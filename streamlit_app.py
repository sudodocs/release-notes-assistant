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

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Release Notes Assistant", layout="wide")

# --- Helper functions ---
@st.cache_data(ttl=3600)
def load_knowledge_base(url):
    """Fetches and loads a JSON knowledge base from a user-provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading Knowledge Base from URL. Please check the URL and file format. Details: {e}")
        return None

def get_prompt(kb, template_name, **kwargs):
    """Safely gets and formats a prompt from the knowledge base."""
    template = kb.get("prompt_templates", {}).get(template_name, "")
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template

def call_ai_provider(prompt, api_key, provider, model_name="gpt-4o", hf_model_id=None, expect_json=False):
    """Calls the selected AI provider and returns the response text."""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash') # Using stable version
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
                st.warning("Hugging Face model ID is required.")
                return ""
            client = InferenceClient(token=api_key)
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=1024)
            return response.strip()
            
    except Exception as e:
        st.error(f"Error calling {provider}: {e}")
        return ""
    return ""


def is_public_api_update(eng_note, kb):
    """
    Checks if an engineering note is a public API update based on keywords and endpoint patterns.
    """
    api_details = kb.get("public_api_details", {})
    keywords = api_details.get("keywords", [])
    patterns = api_details.get("endpoint_patterns", [])
    text_to_check = (eng_note.get("Summary", "") + " " + eng_note.get("Description", "")).lower()

    if any(keyword.lower() in text_to_check for keyword in keywords):
        return True
    
    words_in_text = re.split(r'\s|`|\(|\)|\[|\]', text_to_check)
    for pattern in patterns:
        for word in words_in_text:
            if re.match(pattern, word):
                return True
    return False

def get_api_user_roles(eng_note, kb):
    """
    Determines the applicable user roles for a Public API ticket.
    """
    api_details = kb.get("public_api_details", {})
    role_mapping = api_details.get("role_mapping", {})
    default_roles = api_details.get("default_user_roles", "")
    text_to_check = (eng_note.get("Summary", "") + " " + eng_note.get("Description", "")).lower()
    
    for api_name, roles in role_mapping.items():
        if api_name.lower() in text_to_check:
            return roles
    return default_roles

def convert_md_to_rst(md_text, release_version):
    """
    Converts a Markdown string to a reStructuredText string, creating cross-references
    only for feature categories and a single top-level "Bug Fixes" section.
    """
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


# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'final_report_md' not in st.session_state: st.session_state.final_report_md = None
if 'final_report_rst' not in st.session_state: st.session_state.final_report_rst = None

with st.expander("⚙️ **Configuration**", expanded=True):
    st.info("Please select your AI provider, provide an API key, the URL to your knowledge base, and the release details.")
    ai_provider = st.selectbox("Choose AI Provider", ["OpenAI", "Google Gemini", "Hugging Face"])
    api_key_label = "API Key"
    if ai_provider == "Hugging Face":
        api_key_label = "Hugging Face User Access Token"
    api_key = st.text_input(f"Enter your {api_key_label}", type="password")
    hf_model_id = None
    if ai_provider == "Hugging Face":
        hf_model_id = st.text_input("Enter Hugging Face Model ID", help="e.g., mistralai/Mistral-7B-Instruct-v0.2")

    st.markdown("---")
    kb_url = st.text_input("Knowledge Base URL", placeholder="https://example.com/path/to/your/knowledge_base.json")
    col1, col2, col3 = st.columns(3)
    with col1: release_version = st.text_input("Release Version", "2025.3.1")
    with col2: build_number = st.text_input("Build Number", "2409")
    with col3: release_date = st.text_input("Release Date", "September 28, 2025")

KNOWLEDGE_BASE = load_knowledge_base(kb_url) if kb_url else None

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
                st.error("Please provide an API Key and a valid Knowledge Base URL.")
            else:
                all_dfs = {"Epics": pd.read_csv(epics_csv).fillna(''), "Stories": pd.read_csv(stories_csv).fillna(''), "Bugs": pd.read_csv(bugs_csv).fillna(''), "Escalations": pd.read_csv(escalations_csv).fillna('')}
                progress_bar = st.progress(0)
                total_rows = sum(len(df) for df in all_dfs.values())
                processed_rows, public_items_raw = 0, []

                for name, df in all_dfs.items():
                    for index, row in df.iterrows():
                        processed_rows += 1
                        progress_bar.progress(processed_rows / total_rows, text=f"Processing {name}: {row.get('Summary', '')[:30]}...")
                        eng_note = row.to_dict()

                        # --- [MODIFIED] Added more robust outer try/except block ---
                        try:
                            publicity_prompt = get_prompt(KNOWLEDGE_BASE, 'classifier_publicity', summary=eng_note.get("Summary", ""), issue_type=eng_note.get("Issue Type", ""))
                            publicity_response = call_ai_provider(publicity_prompt, api_key, ai_provider, hf_model_id=hf_model_id)
                            
                            if "PUBLIC" in publicity_response.upper():
                                deployment_prompt = get_prompt(KNOWLEDGE_BASE, 'classifier_deployment', summary=eng_note.get("Summary", ""))
                                eng_note['Deployment'] = call_ai_provider(deployment_prompt, api_key, ai_provider, hf_model_id=hf_model_id)

                                if is_public_api_update(eng_note, KNOWLEDGE_BASE):
                                    eng_note['Category'] = 'Public APIs'
                                else:
                                    categorizer_prompt = get_prompt(KNOWLEDGE_BASE, 'categorizer', company_name=KNOWLEDGE_BASE['company_name'], categories_json=json.dumps(KNOWLEDGE_BASE['product_categories'], indent=2), summary=eng_note.get("Summary", ""), description=(eng_note.get("Description", "") or "")[:300])
                                    cat_response_text = call_ai_provider(categorizer_prompt, api_key, ai_provider, hf_model_id=hf_model_id, expect_json=(ai_provider == "OpenAI"))
                                    
                                    # --- [MODIFIED] Added robust check for JSON response ---
                                    if cat_response_text:
                                        try:
                                            # Clean the response text in case of markdown code blocks
                                            clean_text = re.sub(r'```json\s*|\s*```', '', cat_response_text)
                                            cat_json = json.loads(clean_text)
                                            eng_note['Category'] = cat_json.get('category', 'Other')
                                        except json.JSONDecodeError:
                                            st.warning(f"Failed to decode JSON for '{eng_note.get('Summary')}'. Assigning 'Other'. Response: '{cat_response_text}'")
                                            eng_note['Category'] = 'Other'
                                    else:
                                        st.warning(f"Received empty categorization response for '{eng_note.get('Summary')}'. Assigning 'Other'.")
                                        eng_note['Category'] = 'Other'

                                public_items_raw.append(eng_note)
                        except Exception as e:
                            st.warning(f"Could not process {eng_note.get('Summary')}: {e}")

                df_public = pd.DataFrame(public_items_raw).fillna('')
                df_public['Include'] = True
                public_epic_keys = set(df_public[df_public['Issue Type'] == 'Epic']['Key'])
                df_public['Include'] = df_public.apply(lambda row: False if row['Issue Type'] == 'Story' and row.get('parent') in public_epic_keys else True, axis=1)
                st.session_state.processed_data = df_public
                st.success(f"Triage complete. Found {len(df_public)} potentially public items for your review.")

if st.session_state.processed_data is not None and KNOWLEDGE_BASE:
    with st.container(border=True):
        st.header("Step 2: Review and Approve Items")
        st.warning("Uncheck items to exclude them. You can also correct the AI-suggested Deployment and Category.")
        
        edited_df = st.data_editor(st.session_state.processed_data, column_config={
                "Include": st.column_config.CheckboxColumn("Include?", default=True),
                "Deployment": st.column_config.SelectboxColumn("Deployment", options=["Both", "Cloud Only", "On-Premise Only"], required=True),
                "Category": st.column_config.SelectboxColumn("Category", options=list(KNOWLEDGE_BASE['product_categories'].keys()) + ['Other'], required=True)},
            disabled=["Key", "Summary", "Issue Type", "parent", "Description"], height=400, use_container_width=True)
        
        approved_df = edited_df[edited_df['Include']]
        st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

        if st.button("2️⃣ Generate Document for Approved Items", type="primary", use_container_width=True):
            if not api_key: st.error("Please enter your API key.")
            else:
                features_by_category, bugs_by_category = defaultdict(list), defaultdict(list)
                progress_bar = st.progress(0, text="Writing Final Notes...")

                for i, (_, row) in enumerate(approved_df.iterrows()):
                    progress_bar.progress((i + 1) / len(approved_df), text=f"Writing: {row.get('Summary', '')[:30]}...")
                    eng_note, style = row.to_dict(), KNOWLEDGE_BASE['writing_style_guide']
                    issue_type, category = eng_note.get("Issue Type", "Feature").lower(), eng_note.get('Category', 'Other')
                    task_instruction = style['bug_fix_writing']['instruction'] if "bug" in issue_type or "escalation" in issue_type else style['feature_enhancement_writing']['instruction']
                    user_roles = get_api_user_roles(eng_note, KNOWLEDGE_BASE) if category == "Public APIs" else "Not Applicable"

                    writer_prompt = get_prompt(KNOWLEDGE_BASE, 'writer', company_name=KNOWLEDGE_BASE['company_name'], category=category,
                        professional_tone_rule=style['professional_tone_rule'], terminology_rules_json=json.dumps(style['terminology_rules']),
                        category_specific_instruction=style['category_specific_rules'].get(category, ""),
                        note_json=json.dumps(eng_note, indent=2), user_roles=user_roles, task_instruction=task_instruction)

                    try:
                        suggestion = call_ai_provider(writer_prompt, api_key, ai_provider, hf_model_id=hf_model_id)
                        final_note_text = suggestion
                        deployment_type = row.get('Deployment', 'Both')
                        deployment_map = KNOWLEDGE_BASE.get('deployment_text_mapping', {})

                        if ("bug" in issue_type or "escalation" in issue_type) and (deployment_type in ["Cloud Only", "Both"]):
                            bug_suffix = deployment_map.get('bug_fix_cloud_suffix', '')
                            if bug_suffix: final_note_text = f"{suggestion} {bug_suffix}"
                        elif not ("bug" in issue_type or "escalation" in issue_type):
                            feature_text = deployment_map.get(deployment_type, '')
                            if feature_text: final_note_text += f"\n\n*{feature_text}*"

                        if "bug" in issue_type or "escalation" in issue_type:
                            bugs_by_category[category if category != "Other" else "Other Fixes"].append(final_note_text)
                        else:
                            features_by_category[category].append(final_note_text)
                    except Exception as e:
                        st.warning(f"Could not write note for {row.get('Summary')}: {e}")

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
                    if "Other Fixes" in bugs_by_category: bug_cat_order.append("Other Fixes")
                    for cat_key in bug_cat_order:
                        notes = "\n".join([f"- {note}" if not note.strip().startswith('-') else note for note in bugs_by_category[cat_key]])
                        report_parts.append(f"\n### {cat_key}\n{notes}")
                
                st.session_state.final_report_md = "\n".join(report_parts)
                st.session_state.final_report_rst = convert_md_to_rst(st.session_state.final_report_md, release_version)
                st.success("✅ Release notes document generated successfully!")

if st.session_state.final_report_md:
    with st.container(border=True):
        st.header("Step 3: Download Your Reports")
        st.markdown("### Preview (Markdown)")
        st.markdown(st.session_state.final_report_md)
        
        dl_cols = st.columns(2)
        dl_cols[0].download_button(label="📥 Download Markdown (.md)", data=st.session_state.final_report_md.encode('utf-8'),
            file_name=f"Release_Notes_{release_version}.md", mime="text/markdown", type="primary", use_container_width=True)
        if st.session_state.final_report_rst:
            dl_cols[1].download_button(label="📥 Download RST (.rst)", data=st.session_state.final_report_rst.encode('utf-8'),
                file_name=f"Release_Notes_{release_version}.rst", mime="text/x-rst", type="secondary", use_container_width=True)
