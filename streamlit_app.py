import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime
from collections import defaultdict
import fitz  # PyMuPDF
import io

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

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant")

# Initialize session state
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'final_report' not in st.session_state: st.session_state.final_report = None

with st.expander("⚙️ **Configuration**", expanded=True):
    st.info("Please provide your API key, the URL to your knowledge base, and the release details.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password", placeholder="Enter your OpenAI API Key")
    # Alation-specific KB URL provided as a default example
    kb_url = st.text_input("Knowledge Base URL", "https://raw.githubusercontent.com/mrsauravs/release-notes-assistant/refs/heads/main/knowledge_base.json")
    
    col1, col2, col3 = st.columns(3)
    with col1: release_version = st.text_input("Release Version", "2025.3.1")
    with col2: build_number = st.text_input("Build Number", "2409")
    with col3: release_date = st.text_input("Release Date", "September 28, 2025")

# Load the knowledge base once
KNOWLEDGE_BASE = load_knowledge_base(kb_url) if kb_url and "github" in kb_url else None

with st.container(border=True):
    st.header("Step 1: Upload Your Content Files")
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
        
        edited_df = st.data_editor(
            st.session_state.processed_data,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include?", default=True),
                "Deployment": st.column_config.SelectboxColumn("Deployment", options=["Both", "Cloud Only", "On-Premise Only"], required=True),
                "Category": st.column_config.SelectboxColumn("Category", options=list(KNOWLEDGE_BASE['product_categories'].keys()), required=True)
            },
            disabled=["Key", "Summary", "Issue Type", "parent", "Description"],
            height=400, use_container_width=True
        )
        
        approved_df = edited_df[edited_df['Include']]
        st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

        if st.button("2️⃣ Generate Document for Approved Items", type="primary", use_container_width=True):
            if not api_key: st.error("Please enter your OpenAI API key.")
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

                    if "bug" in issue_type or "escalation" in issue_type: task_instruction = style['bug_fix_writing']['instruction']
                    else: task_instruction = style['feature_enhancement_writing']['instruction']

                    writer_prompt = get_prompt(KNOWLEDGE_BASE, 'writer',
                        company_name=KNOWLEDGE_BASE['company_name'],
                        category=category,
                        professional_tone_rule=style['professional_tone_rule'],
                        terminology_rules_json=json.dumps(style['terminology_rules']),
                        cloud_instruction=f"If the deployment type is 'Cloud Only', you MUST add the suffix '{KNOWLEDGE_BASE['cloud_native_identifier']['suffix_to_add']}' at the very end of the note, after the Jira key.",
                        category_specific_instruction=style['category_specific_rules'].get(category, ""),
                        note_json=json.dumps(eng_note, indent=2),
                        task_instruction=task_instruction
                    )
                    
                    try:
                        writer_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                        suggestion = writer_response.choices[0].message.content.strip()
                        
                        if "bug" in issue_type or "escalation" in issue_type:
                            final_category = "Other Fixes" if category == "Other" else category
                            bugs_by_category[final_category].append(suggestion)
                        else:
                            if category == "Other":
                                try: final_category = suggestion.split('\n')[0].replace('**', '').strip()
                                except IndexError: final_category = eng_note.get('Summary', 'Uncategorized Feature')
                            else: final_category = category
                            features_by_category[final_category].append(suggestion)
                    except Exception as e:
                        st.warning(f"Could not write note for {row.get('Summary')}: {e}")

                # Document Assembly
                main_title = f"# Release {release_version} (Build {build_number})"
                date_subtitle = f"_{release_date}_"
                report_parts = [main_title, date_subtitle]
                
                kb_sections = KNOWLEDGE_BASE['release_structure']['main_sections']
                
                if features_by_category:
                    report_parts.append(f"\n\n**{kb_sections['features']}**\n")
                    for cat_key in KNOWLEDGE_BASE['product_categories']:
                        if cat_key in features_by_category:
                            report_parts.append(f"\n### {cat_key}\n")
                            report_parts.append("\n\n".join(features_by_category.pop(cat_key)))
                    for cat_key, notes in features_by_category.items():
                        report_parts.append(f"\n### {cat_key}\n")
                        report_parts.append("\n\n".join(notes))

                if bugs_by_category:
                    report_parts.append(f"\n\n**{kb_sections['bugs']}**\n")
                    for cat_key in KNOWLEDGE_BASE['product_categories']:
                         if cat_key in bugs_by_category:
                            report_parts.append(f"\n### {cat_key}\n")
                            report_parts.append("\n".join(bugs_by_category.pop(cat_key)))
                    if "Other Fixes" in bugs_by_category:
                         report_parts.append(f"\n### Other Fixes\n")
                         report_parts.append("\n".join(bugs_by_category["Other Fixes"]))
                
                st.session_state.final_report = "\n".join(report_parts)
                st.success("✅ Release notes document generated successfully!")

if st.session_state.final_report:
    with st.container(border=True):
        st.header("Step 3: Download Your Report")
        st.markdown("### Preview")
        st.markdown(st.session_state.final_report)
        st.download_button(
            label="📥 Download Release Notes (.md)",
            data=st.session_state.final_report.encode('utf-8'),
            file_name=f"Release_Notes_{release_version}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True
        )

