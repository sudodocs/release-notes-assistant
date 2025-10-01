import streamlit as st
import pandas as pd
import json
import openai
import google.generativeai as genai
from huggingface_hub import InferenceClient
from datetime import datetime
from collections import defaultdict
import re
import io
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import ipinfo
import sqlite3
import bcrypt

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Release Notes Assistant", layout="wide")

# --- Database Setup ---
def init_db():
    """Initialize SQLite database for users and projects."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        project_name TEXT NOT NULL,
        release_version TEXT NOT NULL,
        markdown TEXT,
        rst TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    conn.commit()
    conn.close()

def register_user(username, password):
    """Register a new user with hashed password."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
        return False
    finally:
        conn.close()

def login_user(username, password):
    """Authenticate a user."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        return user[0]
    return None

def save_project(user_id, project_name, release_version, markdown, rst):
    """Save a project to the database."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO projects (user_id, project_name, release_version, markdown, rst) VALUES (?, ?, ?, ?, ?)",
              (user_id, project_name, release_version, markdown, rst))
    conn.commit()
    conn.close()

def load_projects(user_id):
    """Load all projects for a user."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, project_name, release_version, markdown, rst, created_at FROM projects WHERE user_id = ?", (user_id,))
    projects = c.fetchall()
    conn.close()
    return projects

def log_user_data(username, ip_address):
    """Log user data with hashed username and geolocation."""
    if 'user_data' not in st.session_state:
        st.session_state.user_data = []
    hashed_username = hashlib.sha256(username.encode()).hexdigest()
    try:
        handler = ipinfo.getHandler(st.secrets.get("IPINFO_TOKEN", ""))
        details = handler.getDetails(ip_address)
        location = f"{details.city}, {details.region}, {details.country}" if details.city else "Unknown"
    except:
        location = "Unknown"
    st.session_state.user_data.append({
        'user_id': hashed_username,
        'timestamp': datetime.now().isoformat(),
        'location': location
    })

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_knowledge_base(url):
    """Fetches and loads a JSON knowledge base from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading Knowledge Base: {e}")
        return None

def load_llms_config(url, base_kb):
    """Fetches and merges llms.txt into the KB."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        updates = {}
        for line in content.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                section, subkey = key.split('.') if '.' in key else (key, None)
                if subkey:
                    updates.setdefault(section, {})[subkey] = value
                else:
                    updates[section] = value
        merged_kb = copy.deepcopy(base_kb)
        for k, v in updates.items():
            if isinstance(v, dict) and k in merged_kb:
                merged_kb[k].update(v)
            else:
                merged_kb[k] = v
        return merged_kb
    except Exception as e:
        st.error(f"Error loading llms.txt: {e}")
        return base_kb

def get_prompt(kb, template_name, **kwargs):
    """Formats a prompt from the knowledge base."""
    template = kb.get("prompt_templates", {}).get(template_name, "")
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template

async def async_call_ai_provider(prompt, api_key, provider, model_name="gpt-4o", hf_model_id=None, expect_json=False, retries=3):
    """Async wrapper for AI provider calls with retries."""
    loop = asyncio.get_running_loop()
    for attempt in range(retries):
        try:
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, call_ai_provider, prompt, api_key, provider, model_name, hf_model_id, expect_json)
                return result
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            st.error(f"Failed to call {provider} after {retries} attempts: {e}")
            return ""
    return ""

def call_ai_provider(prompt, api_key, provider, model_name="gpt-4o", hf_model_id=None, expect_json=False):
    """Calls the selected AI provider."""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            completion_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            if expect_json:
                completion_params["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**completion_params)
            return response.choices[0].message.content.strip()
        elif provider == "Hugging Face":
            if not hf_model_id:
                st.warning("Hugging Face model ID required.")
                return ""
            client = InferenceClient(token=api_key)
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=1024)
            return response.strip()
    except Exception as e:
        raise e
    return ""

def is_public_api_update(eng_note, kb, summary_col='Summary', description_col='Description'):
    """Checks if a note is a public API update."""
    api_details = kb.get("public_api_details", {})
    keywords = api_details.get("keywords", [])
    patterns = api_details.get("endpoint_patterns", [])
    text_to_check = (eng_note.get(summary_col, "") + " " + eng_note.get(description_col, "")).lower()
    if any(keyword.lower() in text_to_check for keyword in keywords):
        return True
    words_in_text = re.split(r'\s|`|\(|\)|\[|\]', text_to_check)
    for pattern in patterns:
        for word in words_in_text:
            if re.match(pattern, word):
                return True
    return False

def get_api_user_roles(eng_note, kb, summary_col='Summary', description_col='Description'):
    """Determines user roles for Public API tickets."""
    api_details = kb.get("public_api_details", {})
    role_mapping = api_details.get("role_mapping", {})
    default_roles = api_details.get("default_user_roles", "")
    text_to_check = (eng_note.get(summary_col, "") + " " + eng_note.get(description_col, "")).lower()
    for api_name, roles in role_mapping.items():
        if api_name.lower() in text_to_check:
            return roles
    return default_roles

def convert_md_to_rst(md_text, release_version):
    """Converts Markdown to reStructuredText."""
    toc_entries = []
    features_section_match = re.search(r'(.*?)## Bug Fixes', md_text, re.DOTALL)
    features_section = features_section_match.group(1) if features_section_match else md_text
    feature_headers = re.findall(r"### (.*)", features_section)

    def create_toc_entry(header_title):
        sanitized_title = re.sub(r'[^\w\s-]', '', header_title).strip().lower().replace(' ', '-')
        version_parts = release_version.split('.')
        short_version = f"{version_parts[0]}{version_parts[1]}"
        return f"- :ref:`{header_title} <{sanitized_title}-{short_version}>`"

    for header in feature_headers:
        toc_entries.append(create_toc_entry(header))
    if '## Bug Fixes' in md_text:
        toc_entries.append(create_toc_entry("Bug Fixes"))

    def header_replacer_with_label(match, level):
        header = match.group(1).strip()
        sanitized_title = re.sub(r'[^\w\s-]', '', header).strip().lower().replace(' ', '-')
        version_parts = release_version.split('.')
        short_version = f"{version_parts[0]}{version_parts[1]}"
        label = f".. _{sanitized_title}-{short_version}:"
        underline_char = {2: '-', 3: '~'}.get(level, '')
        return f"\n{label}\n\n{header}\n{underline_char*len(header)}\n"

    def header_replacer_no_label(match, level):
        header = match.group(1).strip()
        underline_char = {2: '-', 3: '~'}.get(level, '')
        return f"\n{header}\n{underline_char*len(header)}\n"

    bug_fixes_heading = "## Bug Fixes"
    if bug_fixes_heading in md_text:
        parts = md_text.split(bug_fixes_heading, 1)
        features_md = parts[0]
        bugs_md = bug_fixes_heading + parts[1]
    else:
        features_md = md_text
        bugs_md = ""

    processed_features = re.sub(r"### (.*)", lambda m: header_replacer_with_label(m, 3), features_md)
    processed_features = re.sub(r"## (.*)", lambda m: header_replacer_with_label(m, 2), processed_features)
    processed_bugs = ""
    if bugs_md:
        bug_lines = bugs_md.split('\n', 1)
        main_bug_header_md = bug_lines[0]
        rest_of_bugs_md = bug_lines[1] if len(bug_lines) > 1 else ""
        processed_main_bug_header = re.sub(r"## (.*)", lambda m: header_replacer_with_label(m, 2), main_bug_header_md)
        processed_rest_of_bugs = re.sub(r"### (.*)", lambda m: header_replacer_no_label(m, 3), rest_of_bugs_md)
        processed_bugs = processed_main_bug_header + processed_rest_of_bugs

    rst_text = processed_features + processed_bugs
    rst_text = re.sub(r"# (.*)", lambda m: f"{m.group(1).strip()}\n{'='*len(m.group(1).strip())}\n", rst_text)
    rst_text = re.sub(r"^\*\*(.*)\*\*$", lambda m: f"{m.group(1).strip()}\n{'^' * len(m.group(1).strip())}\n", rst_text, flags=re.MULTILINE)
    rst_text = re.sub(r"_(.*?)_", r"*\1*", rst_text)

    if toc_entries:
        toc_block = "\nIn this release:\n\n" + "\n".join(toc_entries) + "\n"
        match = re.search(r"(=+|=+\n\n)", rst_text)
        if match:
            insert_pos = match.end()
            rst_text = rst_text[:insert_pos] + toc_block + rst_text[insert_pos:]

    return rst_text.strip()

async def process_single_ticket(row, api_key, provider, hf_model_id, kb, summary_col='Summary', issue_type_col='Issue Type', description_col='Description'):
    """Process a single ticket asynchronously."""
    try:
        eng_note = row.to_dict()
        publicity_prompt = get_prompt(kb, 'classifier_publicity', summary=eng_note.get(summary_col, ""), issue_type=eng_note.get(issue_type_col, ""))
        publicity_response = await async_call_ai_provider(publicity_prompt, api_key, provider, hf_model_id=hf_model_id)
        if "PUBLIC" in publicity_response.upper():
            deployment_prompt = get_prompt(kb, 'classifier_deployment', summary=eng_note.get(summary_col, ""))
            eng_note['Deployment'] = await async_call_ai_provider(deployment_prompt, api_key, provider, hf_model_id=hf_model_id)
            if is_public_api_update(eng_note, kb, summary_col, description_col):
                eng_note['Category'] = 'Public APIs'
            else:
                categorizer_prompt = get_prompt(kb, 'categorizer', company_name=kb['company_name'], categories_json=json.dumps(kb['product_categories'], indent=2), summary=eng_note.get(summary_col, ""), description=(eng_note.get(description_col, "") or "")[:300])
                cat_response_text = await async_call_ai_provider(categorizer_prompt, api_key, provider, hf_model_id=hf_model_id, expect_json=(provider == "OpenAI"))
                if cat_response_text:
                    try:
                        clean_text = re.sub(r'```json\s*|\s*```', '', cat_response_text)
                        cat_json = json.loads(clean_text)
                        eng_note['Category'] = cat_json.get('category', 'Other')
                    except json.JSONDecodeError:
                        st.warning(f"Failed to decode JSON for '{eng_note.get(summary_col)}'. Assigning 'Other'.")
                        eng_note['Category'] = 'Other'
                else:
                    st.warning(f"Empty categorization response for '{eng_note.get(summary_col)}'. Assigning 'Other'.")
                    eng_note['Category'] = 'Other'
            return eng_note
        return None
    except Exception as e:
        st.warning(f"Could not process {eng_note.get(summary_col, 'Unknown')}: {e}")
        return None

async def process_tickets(all_dfs, api_key, provider, hf_model_id, kb):
    """Process all tickets in parallel."""
    tasks = []
    ticket_system = kb.get('ticket_system', {'summary_col': 'Summary', 'issue_type_col': 'Issue Type', 'description_col': 'Description'})
    for name, df in all_dfs.items():
        for _, row in df.iterrows():
            tasks.append(process_single_ticket(row, api_key, provider, hf_model_id, kb, ticket_system['summary_col'], ticket_system['issue_type_col'], ticket_system['description_col']))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if r is not None and not isinstance(r, Exception)]

async def async_write_note(row, api_key, provider, hf_model_id, kb, index, total, progress_bar, summary_col, issue_type_col, description_col):
    """Write a release note asynchronously."""
    try:
        eng_note, style = row.to_dict(), kb['writing_style_guide']
        issue_type, category = eng_note.get(issue_type_col, "Feature").lower(), eng_note.get('Category', 'Other')
        task_instruction = style['bug_fix_writing']['instruction'] if "bug" in issue_type or "escalation" in issue_type else style['feature_enhancement_writing']['instruction']
        user_roles = get_api_user_roles(eng_note, kb, summary_col, description_col) if category == "Public APIs" else "Not Applicable"
        writer_prompt = get_prompt(kb, 'writer', company_name=kb['company_name'], category=category,
            professional_tone_rule=style['professional_tone_rule'], terminology_rules_json=json.dumps(style['terminology_rules']),
            category_specific_instruction=style['category_specific_rules'].get(category, ""),
            note_json=json.dumps(eng_note, indent=2), user_roles=user_roles, task_instruction=task_instruction)
        suggestion = await async_call_ai_provider(writer_prompt, api_key, provider, hf_model_id=hf_model_id)
        final_note_text = suggestion
        deployment_type = row.get('Deployment', 'Both')
        deployment_map = kb.get('deployment_text_mapping', {})
        if ("bug" in issue_type or "escalation" in issue_type) and (deployment_type in ["Cloud Only", "Both"]):
            bug_suffix = deployment_map.get('bug_fix_cloud_suffix', '')
            if bug_suffix:
                final_note_text = f"{suggestion} {bug_suffix}"
        elif not ("bug" in issue_type or "escalation" in issue_type):
            feature_text = deployment_map.get(deployment_type, '')
            if feature_text:
                final_note_text += f"\n\n*{feature_text}*"
        progress_bar.progress((index + 1) / total, text=f"Writing: {row.get(summary_col, '')[:30]}...")
        return category, final_note_text
    except Exception as e:
        st.warning(f"Could not write note for {row.get(summary_col, 'Unknown')}: {e}")
        return None

# --- Main Application Logic ---
init_db()  # Initialize database

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None

if not st.session_state.authenticated:
    st.title("Login to Release Notes Assistant")
    st.info("Log in or register to save and access your release notes projects.")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                user_id = login_user(username, password)
                if user_id:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    log_user_data(username, st.context.get('client_ip', 'Unknown'))
                    st.success(f"Welcome, {username}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            register_button = st.form_submit_button("Register")
            if register_button:
                if register_user(new_username, new_password):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Registration failed. Try a different username.")

else:
    st.title("Intelligent Release Notes Assistant")
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.experimental_rerun()

    # Project Management
    with st.expander("📁 Your Projects", expanded=True):
        projects = load_projects(st.session_state.user_id)
        if projects:
            st.subheader("Saved Projects")
            project_options = [f"{p[2]} - {p[1]} ({p[5]})" for p in projects]
            selected_project = st.selectbox("Select a project to load", [""] + project_options)
            if selected_project:
                project_id = int(selected_project.split(" - ")[1].split(" ")[0])
                for p in projects:
                    if p[0] == project_id:
                        st.session_state.final_report_md = p[3]
                        st.session_state.final_report_rst = p[4]
                        st.success(f"Loaded project: {p[1]}")
                        break
        else:
            st.info("No projects saved yet.")

    # Initialize KB Template
    if 'kb_template' not in st.session_state:
        st.session_state.kb_template = {
            "company_name": "Your Company",
            "ticket_system": {
                "type": "jira",
                "summary_col": "Summary",
                "issue_type_col": "Issue Type",
                "description_col": "Description"
            },
            "deployment_text_mapping": {"Cloud Only": "Cloud", "On-Premise Only": "On-Prem", "Both": "All"},
            "release_structure": {"main_sections": {"features": "New Features", "bugs": "Bug Fixes"}},
            "product_categories": {},
            "writing_style_guide": {
                "professional_tone_rule": "Neutral and professional.",
                "terminology_rules": {},
                "feature_enhancement_writing": {"instruction": "Create a short, descriptive title and benefit-oriented paragraph."},
                "bug_fix_writing": {"instruction": "Write concise bug fix descriptions starting with '- Fixed an issue where'."},
                "category_specific_rules": {}
            },
            "prompt_templates": {
                "classifier_publicity": "Analyze the ticket: {{ \"Summary\": \"{summary}\", \"Issue Type\": \"{issue_type}\" }} Is this PUBLIC or INTERNAL? Respond with one word.",
                "classifier_deployment": "Determine deployment model: Summary: {summary} Respond with: Cloud Only, On-Premise Only, or Both.",
                "categorizer": "Categorize the note for {company_name}. Categories: {categories_json}\nSummary: {summary}\nDescription: {description}\nRespond with JSON: {\"category\": \"value\"}",
                "writer": "Write a release note for {company_name} in category {category}.\nStyle: {professional_tone_rule}\nTerminology: {terminology_rules_json}\nCategory rules: {category_specific_instruction}\nNote: {note_json}\nUser roles: {user_roles}\nTask: {task_instruction}"
            }
        }

    # KB Customization UI
    with st.expander("🛠️ Customize Knowledge Base", expanded=False):
        kb = st.session_state.kb_template.copy()
        kb['company_name'] = st.text_input("Company Name", kb['company_name'])
        st.subheader("Product Categories")
        if 'categories' not in st.session_state:
            st.session_state.categories = []
        for i, cat in enumerate(st.session_state.categories):
            col1, col2, col3 = st.columns([3, 3, 1])
            cat_name = col1.text_input(f"Category Name {i+1}", cat.get('name', ''))
            cat_desc = col2.text_input(f"Description {i+1}", cat.get('description', ''))
            cat_keywords = col3.text_input(f"Keywords {i+1} (comma-separated)", ','.join(cat.get('keywords', [])))
            if st.button("Remove", key=f"remove_cat_{i}"):
                del st.session_state.categories[i]
            st.session_state.categories[i] = {"name": cat_name, "description": cat_desc, "keywords": cat_keywords.split(',')}
        if st.button("Add Category"):
            st.session_state.categories.append({})
        kb['product_categories'] = {cat['name']: {"description": cat['description'], "keywords_and_aliases": cat['keywords']} for cat in st.session_state.categories if cat.get('name')}
        st.json(kb)
        st.download_button("Download KB JSON", data=json.dumps(kb, indent=2), file_name="knowledge_base.json")
        uploaded_kb = st.file_uploader("Upload Existing KB JSON")
        if uploaded_kb:
            st.session_state.kb_template = json.load(uploaded_kb)
            st.success("KB Loaded!")
        if st.button("Use This KB"):
            KNOWLEDGE_BASE = kb
            st.session_state.kb_template = kb

    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        st.info("Select AI provider, provide API key, and enter release details.")
        ai_provider = st.selectbox("Choose AI Provider", ["OpenAI", "Google Gemini", "Hugging Face"])
        api_key = st.secrets.get(f"{ai_provider.upper().replace(' ', '_')}_API_KEY", "")
        if not api_key:
            st.error(f"API key for {ai_provider} not configured. Contact the app administrator.")
            st.stop()
        hf_model_id = None
        if ai_provider == "Hugging Face":
            hf_model_id = st.text_input("Enter Hugging Face Model ID", help="e.g., mistralai/Mistral-7B-Instruct-v0.2")
        st.markdown("---")
        kb_url = st.text_input("Knowledge Base URL", placeholder="https://example.com/knowledge_base.json")
        llms_url = st.text_input("LLMs Config URL (optional)", placeholder="https://example.com/llms.txt")
        col1, col2, col3 = st.columns(3)
        with col1: release_version = st.text_input("Release Version", "2025.3.1")
        with col2: build_number = st.text_input("Build Number", "2409")
        with col3: release_date = st.text_input("Release Date", "September 28, 2025")
        project_name = st.text_input("Project Name", f"Release {release_version}")

    KNOWLEDGE_BASE = load_knowledge_base(kb_url) if kb_url else None
    if llms_url and KNOWLEDGE_BASE:
        KNOWLEDGE_BASE = load_llms_config(llms_url, KNOWLEDGE_BASE)
        st.success("KB updated with llms.txt.")

    # Step 1: Upload
    with st.container(border=True):
        st.header("Step 1: Upload Your Content Files")
        upload_cols = st.columns(4)
        epics_csv = upload_cols[0].file_uploader("1. Epics", type="csv")
        stories_csv = upload_cols[1].file_uploader("2. Stories", type="csv")
        bugs_csv = upload_cols[2].file_uploader("3. Bug Fixes", type="csv")
        escalations_csv = upload_cols[3].file_uploader("4. Support Escalations", type="csv")

        if all([epics_csv, stories_csv, bugs_csv, escalations_csv]):
            if st.button("1️⃣ Triage & Categorize Items", type="primary", use_container_width=True):
                if not api_key or not KNOWLEDGE_BASE:
                    st.error("Please provide a valid Knowledge Base URL.")
                else:
                    with st.spinner("Processing tickets..."):
                        all_dfs = {
                            "Epics": pd.read_csv(epics_csv).fillna(''),
                            "Stories": pd.read_csv(stories_csv).fillna(''),
                            "Bugs": pd.read_csv(bugs_csv).fillna(''),
                            "Escalations": pd.read_csv(escalations_csv).fillna('')
                        }
                        ticket_system = KNOWLEDGE_BASE.get('ticket_system', {})
                        required_cols = [ticket_system.get('summary_col', 'Summary'), ticket_system.get('issue_type_col', 'Issue Type')]
                        for name, df in all_dfs.items():
                            missing_cols = [col for col in required_cols if col not in df.columns]
                            if missing_cols:
                                st.error(f"Missing required columns in {name} CSV: {', '.join(missing_cols)}")
                                st.stop()
                        public_items_raw = asyncio.run(process_tickets(all_dfs, api_key, ai_provider, hf_model_id, KNOWLEDGE_BASE))
                        df_public = pd.DataFrame(public_items_raw).fillna('')
                        df_public['Include'] = True
                        public_epic_keys = set(df_public[df_public[ticket_system.get('issue_type_col', 'Issue Type')] == 'Epic']['Key'])
                        df_public['Include'] = df_public.apply(
                            lambda row: False if row[ticket_system.get('issue_type_col', 'Issue Type')] == 'Story' and row.get('parent') in public_epic_keys else True, axis=1)
                        st.session_state.processed_data = df_public
                        st.success(f"Triage complete. Found {len(df_public)} potentially public items for your review.")

    # Step 2: Review
    if st.session_state.get('processed_data') is not None and KNOWLEDGE_BASE:
        with st.container(border=True):
            st.header("Step 2: Review and Approve Items")
            st.warning("Uncheck items to exclude them. You can also correct the AI-suggested Deployment and Category.")
            edited_df = st.data_editor(
                st.session_state.processed_data,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include?", default=True),
                    "Deployment": st.column_config.SelectboxColumn("Deployment", options=["Both", "Cloud Only", "On-Premise Only"], required=True),
                    "Category": st.column_config.SelectboxColumn("Category", options=list(KNOWLEDGE_BASE['product_categories'].keys()) + ['Other'], required=True)
                },
                disabled=["Key", KNOWLEDGE_BASE.get('ticket_system', {}).get('summary_col', 'Summary'), KNOWLEDGE_BASE.get('ticket_system', {}).get('issue_type_col', 'Issue Type'), "parent", KNOWLEDGE_BASE.get('ticket_system', {}).get('description_col', 'Description')],
                height=400,
                use_container_width=True,
                num_rows='dynamic'
            )
            approved_df = edited_df[edited_df['Include']]
            st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

            if st.button("2️⃣ Generate Document for Approved Items", type="primary", use_container_width=True):
                if not api_key:
                    st.error("API key not configured. Contact the app administrator.")
                else:
                    with st.spinner("Generating release notes..."):
                        features_by_category, bugs_by_category = defaultdict(list), defaultdict(list)
                        progress_bar = st.progress(0, text="Writing Final Notes...")
                        ticket_system = KNOWLEDGE_BASE.get('ticket_system', {})
                        summary_col = ticket_system.get('summary_col', 'Summary')
                        issue_type_col = ticket_system.get('issue_type_col', 'Issue Type')
                        description_col = ticket_system.get('description_col', 'Description')
                        tasks = []
                        for i, (_, row) in enumerate(approved_df.iterrows()):
                            tasks.append(async_write_note(row, api_key, ai_provider, hf_model_id, KNOWLEDGE_BASE, i, len(approved_df), progress_bar, summary_col, issue_type_col, description_col))
                        results = asyncio.run(asyncio.gather(*tasks, return_exceptions=True))
                        for i, result in enumerate(results):
                            if not isinstance(result, Exception):
                                category, note = result
                                if "bug" in approved_df.iloc[i][issue_type_col].lower() or "escalation" in approved_df.iloc[i][issue_type_col].lower():
                                    bugs_by_category[category if category != "Other" else "Other Fixes"].append(note)
                                else:
                                    features_by_category[category].append(note)
                        main_title, date_subtitle = f"# Release {release_version} (Build {build_number})", f"_{release_date}_"
                        report_parts = [main_title, date_subtitle]
                        kb_sections = KNOWLEDGE_BASE['release_structure']['main_sections']
                        if features_by_category:
                            report_parts.append(f"\n\n## {kb_sections['features']}\n")
                            all_feature_cats = list(features_by_category.keys())
                            ordered_cats = [cat for cat in KNOWLEDGE_BASE['product_categories'] if cat in all_feature_cats]
                            remaining_cats = sorted([cat for cat in all_feature_cats if cat not in KNOWLEDGE_BASE['product_categories']])
                            for cat_key in ordered_cats + remaining_cats:
                                notes = "\n\n".join(features_by_category[cat_key])
                                report_parts.append(f"\n### {cat_key}\n{notes}")
                        if bugs_by_category:
                            report_parts.append(f"\n\n## {kb_sections['bugs']}\n")
                            bug_cat_order = [cat for cat in KNOWLEDGE_BASE['product_categories'] if cat in bugs_by_category]
                            if "Other Fixes" in bugs_by_category:
                                bug_cat_order.append("Other Fixes")
                            for cat_key in bug_cat_order:
                                notes = "\n".join([f"- {note}" if not note.strip().startswith('-') else note for note in bugs_by_category[cat_key]])
                                report_parts.append(f"\n### {cat_key}\n{notes}")
                        st.session_state.final_report_md = "\n".join(report_parts)
                        st.session_state.final_report_rst = convert_md_to_rst(st.session_state.final_report_md, release_version)
                        save_project(st.session_state.user_id, project_name, release_version, st.session_state.final_report_md, st.session_state.final_report_rst)
                        st.success(f"✅ Release notes generated and saved as project: {project_name}")

    # Step 3: Download
    if st.session_state.get('final_report_md'):
        with st.container(border=True):
            st.header("Step 3: Download Your Reports")
            st.markdown("### Preview (Markdown)")
            st.markdown(st.session_state.final_report_md)
            dl_cols = st.columns(2)
            dl_cols[0].download_button(
                label="📥 Download Markdown (.md)",
                data=st.session_state.final_report_md.encode('utf-8'),
                file_name=f"Release_Notes_{release_version}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
            if st.session_state.final_report_rst:
                dl_cols[1].download_button(
                    label="📥 Download RST (.rst)",
                    data=st.session_state.final_report_rst.encode('utf-8'),
                    file_name=f"Release_Notes_{release_version}.rst",
                    mime="text/x-rst",
                    type="secondary",
                    use_container_width=True
                )

    # User Stats (Admin Only)
    with st.expander("📊 User Statistics (Admin Only)"):
        if st.session_state.get('user_data'):
            st.write(f"Total Registered Users: {len(set(d['user_id'] for d in st.session_state.user_data))}")
            st.write("Recent Logins:")
            for data in st.session_state.user_data[-10:]:
                st.write(f"User (hashed): {data['user_id'][:8]}..., Time: {data['timestamp']}, Location: {data['location']}")
        else:
            st.info("No user data yet.")
