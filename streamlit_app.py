import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime
from collections import defaultdict
import re # Import regex module for RST conversion
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
    return template.format(**kwargs)

def convert_md_to_rst(md_text, release_version):
    """Converts a Markdown string to a reStructuredText string."""
    rst_text = md_text
    toc_entries = []

    # Helper function to create a valid RST label and TOC entry from a header title
    def create_label_and_toc(header_title):
        # FIXED: More robust sanitization to remove punctuation from labels
        sanitized_title = re.sub(r'[^\w\s-]', '', header_title).strip().lower().replace(' ', '-')
        
        version_parts = release_version.split('.')
        short_version = f"{version_parts[0]}{version_parts[1]}"
        
        label = f".. _{sanitized_title}-{short_version}:"
        toc_entry = f"- :ref:`{header_title} <{sanitized_title}-{short_version}>`"
        return label, toc_entry

    # This function is used by re.sub to replace each Markdown H3 header
    def h3_header_replacer(match):
        header = match.group(1)
        label, toc_entry = create_label_and_toc(header)
        toc_entries.append(toc_entry)
        # FIXED: Added a newline at the end for spacing
        return f"{label}\n\n{header}\n{'~'*len(header)}\n"

    # --- Header Conversion (Order is important) ---

    # Process H3 headers first as they are the most complex
    rst_text = re.sub(r"### (.*)", h3_header_replacer, rst_text)

    # Convert bolded titles on their own line to RST H4 headers
    def h4_header_replacer(match):
        title = match.group(1)
        # FIXED: Added a newline at the end for spacing
        return f"{title}\n{'^' * len(title)}\n"
    rst_text = re.sub(r"^\*\*(.*)\*\*$", h4_header_replacer, rst_text, flags=re.MULTILINE)

    # FIXED: Process H2 and H1 headers *after* H3, in the correct order
    # Convert H2 (## Section) headers
    rst_text = re.sub(r"## (.*)", lambda m: f"{m.group(1)}\n{'-'*len(m.group(1))}\n", rst_text)
    # Convert H1 (# Title) headers
    rst_text = re.sub(r"# (.*)", lambda m: f"{m.group(1)}\n{'='*len(m.group(1))}\n", rst_text)
    
    # Convert _italics_ to *italics* for RST compatibility
    rst_text = re.sub(r"_(.*?)_", r"*\1*", rst_text)

    # Insert the generated Table of Contents after the date line
    if toc_entries:
        toc_block = "In this release:\n\n" + "\n".join(toc_entries)
        parts = rst_text.split('\n', 2)
        if len(parts) > 2:
            rst_text = f"{parts[0]}\n{parts[1]}\n\n{toc_block}\n\n{parts[2]}"

    return rst_text

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant")

# Initialize session state
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'final_report_md' not in st.session_state: st.session_state.final_report_md = None
if 'final_report_rst' not in st.session_state: st.session_state.final_report_rst = None

with st.expander("⚙️ **Configuration**", expanded=True):
    st.info("Please provide your API key, the URL to your knowledge base, and the release details.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password", placeholder="Enter your OpenAI API Key")
    kb_url = st.text_input("Knowledge Base URL", placeholder="https://example.com/path/to/your/knowledge_base.json")

    col1, col2, col3 = st.columns(3)
    with col1: release_version = st.text_input("Release Version", "2025.0.1")
    with col2: build_number = st.text_input("Build Number", "2409")
    with col3: release_date = st.text_input("Release Date", "September 28, 2025")

KNOWLEDGE_BASE = load_knowledge_base(kb_url) if kb_url else None

with st.container(border=True):
    st.header("Step 1: Upload Your Content Files")
    # ... (rest of the file upload logic is unchanged) ...
    col1, col2, col3, col4 = st.columns(4)
    with col1: epics_csv = st.file_uploader("1. Epics", type="csv")
    with col2: stories_csv = st.file_uploader("2. Stories", type="csv")
    with col3: bugs_csv = st.file_uploader("3. Bug Fixes", type="csv")
    with col4: escalations_csv = st.file_uploader("4. Support Escalations", type="csv")

    if epics_csv and stories_csv and bugs_csv and escalations_csv:
        if st.button("1️⃣ Triage & Categorize Items", type="primary", use_container_width=True):
            if not api_key or not KNOWLEDGE_BASE:
                st.error("Please provide both an API Key and a valid Knowledge Base URL in the Configuration section.")
            else:
                # ... (rest of the data processing logic is unchanged) ...
                client = openai.OpenAI(api_key=api_key)
                all_dfs = {"Epics": pd.read_csv(epics_csv).fillna(''), "Stories": pd.read_csv(stories_csv).fillna(''), "Bugs": pd.read_csv(bugs_csv).fillna(''), "Escalations": pd.read_csv(escalations_csv).fillna('')}
                progress_bar = st.progress(0)
                total_rows = sum(len(df) for df in all_dfs.values())
                processed_rows = 0
                public_items_raw = []

                for name, df in all_dfs.items():
                    for index, row in df.iterrows():
                        processed_rows += 1
                        progress_bar.progress(processed_rows / total_rows, text=f"Processing {name}: {row.get('Summary', '')[:30]}...")
                        eng_note = row.to_dict()

                        publicity_prompt = get_prompt(KNOWLEDGE_BASE, 'classifier_publicity', summary=eng_note.get("Summary", ""), issue_type=eng_note.get("Issue Type", ""))
                        try:
                            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": publicity_prompt}], max_tokens=5, temperature=0)
                            if "PUBLIC" in response.choices[0].message.content.strip().upper():
                                deployment_prompt = get_prompt(KNOWLEDGE_BASE, 'classifier_deployment', summary=eng_note.get("Summary", ""))
                                dep_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": deployment_prompt}], max_tokens=10, temperature=0)
                                eng_note['Deployment'] = dep_response.choices[0].message.content.strip()

                                categorizer_prompt = get_prompt(KNOWLEDGE_BASE, 'categorizer', company_name=KNOWLEDGE_BASE['company_name'], categories_json=json.dumps(KNOWLEDGE_BASE['product_categories'], indent=2), summary=eng_note.get("Summary", ""), description=(eng_note.get("Description", "") or "")[:300])
                                cat_response = client.chat.completions.create(model="gpt-4o", response_format={"type": "json_object"}, messages=[{"role": "user", "content": categorizer_prompt}])
                                cat_json = json.loads(cat_response.choices[0].message.content)
                                eng_note['Category'] = cat_json.get('category', 'Other')

                                public_items_raw.append(eng_note)
                        except Exception as e:
                            st.warning(f"Could not process {eng_note.get('Summary')}: {e}")

                df_public = pd.DataFrame(public_items_raw).fillna('')
                df_public['Include'] = True
                public_epic_keys = set(df_public[df_public['Issue Type'] == 'Epic']['Key'])
                df_public['Include'] = df_public.apply(lambda row: False if row['Issue Type'] == 'Story' and row['parent'] in public_epic_keys else True, axis=1)

                st.session_state.processed_data = df_public
                st.success(f"Triage complete. Found {len(df_public)} potentially public items for your review.")


if st.session_state.processed_data is not None and KNOWLEDGE_BASE:
    with st.container(border=True):
        st.header("Step 2: Review and Approve Items")
        st.warning("Uncheck items to exclude them. You can also correct the AI-suggested Deployment and Category.")
        
        # ... (data_editor logic is unchanged) ...
        edited_df = st.data_editor(
            st.session_state.processed_data,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include?", default=True),
                "Deployment": st.column_config.SelectboxColumn("Deployment", options=["Both", "Cloud Only", "On-Premise Only"], required=True),
                "Category": st.column_config.SelectboxColumn("Category", options=list(KNOWLEDGE_BASE['product_categories'].keys()) + ['Other'], required=True)
            },
            disabled=["Key", "Summary", "Issue Type", "parent", "Description"],
            height=400, use_container_width=True
        )
        approved_df = edited_df[edited_df['Include']]
        st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

       if st.button("2️⃣ Generate Document for Approved Items", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenAI API key.")
            else:
                client = openai.OpenAI(api_key=api_key)
                features_by_category = defaultdict(list)
                bugs_by_category = defaultdict(list)

                progress_bar = st.progress(0, text="Writing Final Notes...")
                total_to_write = len(approved_df)

                for i, (index, row) in enumerate(approved_df.iterrows()):
                    progress_bar.progress((i + 1) / total_to_write, text=f"Writing: {row.get('Summary', '')[:30]}...")
                    eng_note = row.to_dict()

                    style = KNOWLEDGE_BASE['writing_style_guide']
                    issue_type = eng_note.get("Issue Type", "Feature").lower()
                    category = eng_note.get('Category', 'Other')

                    if "bug" in issue_type or "escalation" in issue_type:
                        task_instruction = style['bug_fix_writing']['instruction']
                    else:
                        task_instruction = style['feature_enhancement_writing']['instruction']
                    
                    # MODIFIED: Removed the old 'cloud_instruction' from the prompt
                    writer_prompt = get_prompt(KNOWLEDGE_BASE, 'writer',
                        company_name=KNOWLEDGE_BASE['company_name'],
                        category=category,
                        professional_tone_rule=style['professional_tone_rule'],
                        terminology_rules_json=json.dumps(style['terminology_rules']),
                        category_specific_instruction=style['category_specific_rules'].get(category, ""),
                        note_json=json.dumps(eng_note, indent=2),
                        task_instruction=task_instruction
                    )

                    try:
                        writer_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                        suggestion = writer_response.choices[0].message.content.strip()

                        # --- NEW: Logic to append Jira Key and Deployment Text ---
                        jira_key = row.get('Key', '')
                        deployment_type = row.get('Deployment', 'Both')
                        deployment_map = KNOWLEDGE_BASE.get('deployment_text_mapping', {})
                        
                        final_note_parts = [suggestion] # Start with AI-written text

                        # Part 1: Description and Jira Key
                        if jira_key:
                            final_note_parts.append(f"({jira_key})")
                        
                        # Part 2: Bug Fix Suffix (Cloud Only)
                        if ("bug" in issue_type or "escalation" in issue_type) and (deployment_type in ["Cloud Only", "Both"]):
                            bug_suffix = deployment_map.get('bug_fix_cloud_suffix', '')
                            if bug_suffix:
                                final_note_parts.append(bug_suffix)
                        
                        final_note_text = " ".join(final_note_parts)

                        # Part 3: Feature Deployment Text (On a new line)
                        if not ("bug" in issue_type or "escalation" in issue_type):
                            feature_text = deployment_map.get(deployment_type, '')
                            if feature_text:
                                final_note_text += f"\n\n*{feature_text}*"
                        # --- END: New Logic ---

                        if "bug" in issue_type or "escalation" in issue_type:
                            final_category = "Other Fixes" if category == "Other" else category
                            bugs_by_category[final_category].append(final_note_text)
                        else:
                            if category == "Other":
                                try:
                                    final_category = suggestion.split('\n')[0].replace('**', '').strip()
                                except IndexError:
                                    final_category = eng_note.get('Summary', 'Uncategorized Feature')
                            else:
                                final_category = category
                            features_by_category[final_category].append(final_note_text)
                    except Exception as e:
                        st.warning(f"Could not write note for {row.get('Summary')}: {e}")
                
                # Document Assembly
                main_title = f"# Release {release_version} (Build {build_number})"
                date_subtitle = f"_{release_date}_"
                report_parts = [main_title, date_subtitle]
                kb_sections = KNOWLEDGE_BASE['release_structure']['main_sections']

                # ... (Feature assembly logic with redundancy check is unchanged) ...
                if features_by_category:
                    report_parts.append(f"\n\n## {kb_sections['features']}\n")
                    def normalize_text(text):
                        return text.replace('####', '').replace('###', '').replace('**', '').strip().lower()
                    all_feature_categories = list(features_by_category.keys())
                    ordered_categories = [cat for cat in KNOWLEDGE_BASE['product_categories'] if cat in all_feature_categories]
                    remaining_categories = [cat for cat in all_feature_categories if cat not in KNOWLEDGE_BASE['product_categories']]
                    final_category_order = ordered_categories + remaining_categories
                    for cat_key in final_category_order:
                        if cat_key not in features_by_category: continue
                        notes = features_by_category[cat_key]
                        report_parts.append(f"\n### {cat_key}\n")
                        processed_notes = []
                        normalized_cat_key = normalize_text(cat_key)
                        for note in notes:
                            lines = note.strip().split('\n')
                            if not lines: continue
                            title_line = lines[0]
                            normalized_title = normalize_text(title_line)
                            if normalized_cat_key == normalized_title:
                                processed_notes.append("\n".join(lines[1:]).strip())
                            else:
                                processed_notes.append(note)
                        report_parts.append("\n\n".join(processed_notes))

                # ... (Bug fix assembly logic is unchanged) ...
                if bugs_by_category:
                    report_parts.append(f"\n\n## {kb_sections['bugs']}\n")
                    for cat_key in KNOWLEDGE_BASE['product_categories']:
                         if cat_key in bugs_by_category:
                            report_parts.append(f"\n### {cat_key}\n")
                            report_parts.append("\n".join([note if note.strip().startswith('-') else f"- {note}" for note in bugs_by_category.pop(cat_key)]))
                    for cat_key, notes in sorted(bugs_by_category.items()):
                        report_parts.append(f"\n### {cat_key}\n")
                        report_parts.append("\n".join([note if note.strip().startswith('-') else f"- {note}" for note in notes]))

                # --- NEW: Generate both formats and store in session state ---
                final_md_report = "\n".join(report_parts)
                st.session_state.final_report_md = final_md_report
                st.session_state.final_report_rst = convert_md_to_rst(final_md_report, release_version)
                
                st.success("✅ Release notes document generated successfully!")

if st.session_state.final_report_md:
    with st.container(border=True):
        st.header("Step 3: Download Your Reports")
        st.markdown("### Preview (Markdown)")
        st.markdown(st.session_state.final_report_md)
        
        # --- NEW: Dual download buttons ---
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Markdown (.md)",
                data=st.session_state.final_report_md.encode('utf-8'),
                file_name=f"Release_Notes_{release_version}.md",
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="📥 Download RST (.rst)",
                data=st.session_state.final_report_rst.encode('utf-8'),
                file_name=f"Release_Notes_{release_version}.rst",
                mime="text/x-rst",
                type="secondary",
                use_container_width=True
            )
