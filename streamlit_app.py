import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime
from collections import defaultdict
import fitz  # PyMuPDF
import io

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Release Notes Assistant 🚀", layout="wide")

# --- Embedded Knowledge Base ---
KNOWLEDGE_BASE = {
  "company_name": "Alation",
  "release_structure": { "section_order": [ "New Features", "Enhancements", "Bug Fixes" ] },
  "product_categories": [ "Content Experience", "Curation", "Compose", "Alation Analytics", "Search", "Open Connector Framework", "Lineage", "Admin Settings", "Platform", "Public APIs", "Other" ],
  "writing_style_guide": {
    "professional_tone_rule": "Adopt a neutral, professional tone (similar to the Microsoft Style Guide). DO NOT use overly enthusiastic phrases like 'We are excited to announce...'. State the facts directly.",
    "terminology_rules": { "Neo": "New User Experience", "New UI": "New User Experience", "Classic UI": "Classic User Experience", "old UI": "Classic User Experience" },
    "feature_enhancement_writing": { "instruction": "First, create a short, descriptive title from the 'Summary'. The title should be bolded. On the next line, write a clear, benefit-oriented paragraph. At the end, add the Jira key in parentheses." },
    "bug_fix_writing": { "instruction": "Generate a single, concise sentence for a bullet point starting with 'Fixed an issue where...'. At the end, add the Jira key in parentheses." }
  },
  "cloud_native_identifier": { "suffix_to_add": "(Alation Cloud Service)" }
}

# --- Helper functions ---
def build_classifier_prompt(note, type):
    if type == 'publicity':
        return f"""Analyze the ticket data based on its **ultimate outcome** for the end-user. Backend work is PUBLIC if the result is a new capability, a noticeable performance improvement, or a fixed bug.
        Ticket Data: {{ "Summary": "{note.get("Summary", "")}", "Issue Type": "{note.get("Issue Type", "")}" }}
        Is this change PUBLIC or INTERNAL? Respond with a single word."""
    elif type == 'deployment':
        return f"""You are an Alation Release Manager. Analyze the engineering note to determine its deployment model.
        - **Cloud Only**: Mentions 'Alation Cloud Service', 'ACS', or is a feature that cannot exist on-prem.
        - **On-Premise Only**: Mentions 'server installation', 'update pack', 'RPM', or is a feature for customer-managed environments.
        - **Both**: If it mentions both, or is a general product feature, classify it as 'Both'.
        **Ticket Data:**
        Summary: {note.get("Summary", "")}
        Description: {(note.get("Description", "") or "")[:400]}
        Your response must be one of: Cloud Only, On-Premise Only, or Both."""

def build_categorizer_prompt(note, categories):
    return f"""Categorize this ticket into one of the following official product areas. Choose the single best fit.
    **Valid Categories:** {', '.join(categories)}
    **Ticket Summary:** {note.get("Summary", "")}
    Response must be only the category name."""

def build_release_prompt(kb, note, category, deployment_type):
    style = kb['writing_style_guide']
    issue_type = note.get("Issue Type", "Feature").lower()

    if "bug" in issue_type or "escalation" in issue_type:
        task_instruction = f"**Task:** Write a single sentence for a Markdown bullet point. {style['bug_fix_writing']['instruction']}"
    else:
        task_instruction = f"**Task:** Write the release note. {style['feature_enhancement_writing']['instruction']}"
    
    cloud_suffix = kb['cloud_native_identifier']['suffix_to_add']
    cloud_instruction = f"If the deployment type is 'Cloud Only', you MUST add the suffix '{cloud_suffix}' at the very end of the note, after the Jira key. For all other deployment types, do not add any suffix."

    prompt = f"""
    You are a Principal Technical Writer at {kb['company_name']}, following the Microsoft Style Guide. Convert the raw engineering note for the '{category}' category into a formal release note.

    **CRITICAL WRITING RULES:**
    1.  **Professional Tone:** {style['professional_tone_rule']}
    2.  **Terminology Substitution:** Replace these internal names: {json.dumps(style['terminology_rules'])}.
    3.  **Sanitize Output:** Remove all internal jargon.
    4.  **Append Jira Key:** At the end of the note, you MUST append the 'Key' (the Jira key), enclosed in parentheses.
    5.  **Cloud Suffix:** {cloud_instruction}

    **Raw Engineering Note:** ```json\n{json.dumps(note, indent=2)}\n```
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title(f"Intelligent Release Notes Assistant 🚀")

# Initialize session state
if 'classified_data' not in st.session_state: st.session_state.classified_data = None
if 'final_report' not in st.session_state: st.session_state.final_report = None

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The knowledge base is embedded in the application.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    release_version = st.text_input("Enter Release Version (e.g., 2025.3.1)", "2025.3.1")

st.header("Step 1: Upload Your Content Files")
col1, col2, col3, col4 = st.columns(4)
with col1: epics_csv = st.file_uploader("1. Epics", type="csv")
with col2: stories_csv = st.file_uploader("2. Stories", type="csv")
with col3: bugs_csv = st.file_uploader("3. Bug Fixes", type="csv")
with col4: escalations_csv = st.file_uploader("4. Support Escalations", type="csv")

if epics_csv and stories_csv and bugs_csv and escalations_csv:
    if st.button("1️⃣ Classify All Items"):
        st.session_state.classified_data = None
        st.session_state.final_report = None
        if not api_key: st.error("Please enter your OpenAI API key.")
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
                    progress_bar.progress(processed_rows / total_rows, text=f"Triaging {name}: {row.get('Summary', '')[:30]}...")
                    eng_note = row.to_dict()
                    
                    # AI Publicity Check
                    publicity_prompt = build_classifier_prompt(eng_note, 'publicity')
                    try:
                        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": publicity_prompt}], max_tokens=5, temperature=0)
                        classification = response.choices[0].message.content.strip().upper()
                        if "PUBLIC" in classification:
                            # AI Deployment Check
                            deployment_prompt = build_classifier_prompt(eng_note, 'deployment')
                            dep_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": deployment_prompt}], max_tokens=10, temperature=0)
                            eng_note['Deployment'] = dep_response.choices[0].message.content.strip()
                            public_items_raw.append(eng_note)
                    except Exception as e:
                        st.warning(f"Could not classify {eng_note.get('Summary')}: {e}")
            
            df_public = pd.DataFrame(public_items_raw).fillna('')
            df_public['Include'] = True
            public_epic_keys = set(df_public[df_public['Issue Type'] == 'Epic']['Key'])
            df_public['Include'] = df_public.apply(lambda row: False if row['Issue Type'] == 'Story' and row['parent'] in public_epic_keys else True, axis=1)
            
            st.session_state.classified_data = df_public
            st.success(f"Triage complete. Found {len(df_public)} potentially public items for your review.")

if st.session_state.classified_data is not None:
    st.header("Step 2: Review and Approve Items")
    st.warning("Uncheck items to exclude them. You can also correct the AI-suggested Deployment type.")
    
    edited_df = st.data_editor(
        st.session_state.classified_data,
        column_config={
            "Include": st.column_config.CheckboxColumn("Include?", default=True),
            "Deployment": st.column_config.SelectboxColumn(
                "Deployment", options=["Both", "Cloud Only", "On-Premise Only"], required=True
            )
        },
        disabled=["Key", "Summary", "Issue Type", "parent", "Description"],
        height=400, use_container_width=True
    )
    
    approved_df = edited_df[edited_df['Include']]
    st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

    if st.button("2️⃣ Generate Document for Approved Items"):
        if not api_key: st.error("Please enter your OpenAI API key.")
        else:
            client = openai.OpenAI(api_key=api_key)
            final_notes_by_category = defaultdict(lambda: defaultdict(list))
            progress_bar = st.progress(0, text="Categorizing and Writing Notes...")
            total_to_write = len(approved_df)
            
            for i, (index, row) in enumerate(approved_df.iterrows()):
                progress_bar.progress((i + 1) / total_to_write, text=f"Processing: {row.get('Summary', '')[:30]}...")
                eng_note = row.to_dict()
                
                # AI Categorizer Step
                categorizer_prompt = build_categorizer_prompt(eng_note, KNOWLEDGE_BASE['product_categories'])
                cat_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": categorizer_prompt}], max_tokens=20, temperature=0)
                category = cat_response.choices[0].message.content.strip()
                if category not in KNOWLEDGE_BASE['product_categories']: category = "Other"

                # AI Writer Step
                writer_prompt = build_release_prompt(KNOWLEDGE_BASE, eng_note, category, eng_note['Deployment'])
                writer_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                suggestion = writer_response.choices[0].message.content.strip()
                
                issue_type = eng_note.get("Issue Type", "Feature").lower()
                if "bug" in issue_type or "escalation" in issue_type: final_notes_by_category[category]["Bug Fixes"].append(suggestion)
                else: final_notes_by_category[category]["New Features"].append(suggestion)

            # Assemble Final Document
            month_year = datetime.now().strftime('%B %Y')
            report_parts = [f"# Release {release_version}", f"_{month_year}_"]
            for category in KNOWLEDGE_BASE['product_categories']:
                if category in final_notes_by_category:
                    report_parts.append(f"\n\n### {category}")
                    for section in KNOWLEDGE_BASE['release_structure']['section_order']:
                        if final_notes_by_category[category].get(section):
                            report_parts.append(f"\n\n**{section}**\n")
                            if section == "Bug Fixes": report_parts.append("\n".join(final_notes_by_category[category][section]))
                            else: report_parts.append("\n\n".join(final_notes_by_category[category][section]))
            
            st.session_state.final_report = "\n".join(report_parts)
            st.success("✅ Release notes document generated successfully!")

if st.session_state.final_report:
    st.header("Step 3: Download Report")
    st.markdown("### Preview")
    st.markdown(st.session_state.final_report)
    st.download_button(
        label="📥 Download Release Notes (.md)",
        data=st.session_state.final_report.encode('utf-8'),
        file_name=f"Release_Notes_{release_version}.md",
        mime="text/markdown",
    )
