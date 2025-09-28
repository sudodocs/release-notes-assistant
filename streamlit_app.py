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

# --- CORRECTED: Simplified CSS for a clean and robust UI ---
st.markdown("""
<style>
    /* Main font and background */
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }
    .stApp {
        background-color: #f0f2f6;
    }

    /* Headers for clear hierarchy */
    h1, h2, h3 {
        font-weight: 600;
        color: #1E293B; /* Darker text for better contrast */
    }

    /* Primary CTA button styling */
    .stButton > button[kind="primary"] {
        background-color: #0068c9;
        font-weight: 600;
        border-radius: 8px;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #00509a;
    }

</style>
""", unsafe_allow_html=True)


# --- Embedded Knowledge Base ---
KNOWLEDGE_BASE = {
  "company_name": "Alation",
  "release_structure": { "section_order": [ "New Features", "Enhancements", "Bug Fixes" ] },
  "product_categories": {
    "Content Experience": {"description": "Features related to how users create, view, and manage documentation and curated content.", "keywords_and_aliases": ["Articles", "DocHubs", "Document Hubs", "Glossary", "Lexicon"]},
    "Curation": {"description": "Features for data stewards to enrich the catalog, such as applying tags, flags, and custom fields.", "keywords_and_aliases": ["Stewardship", "Tags", "Trust Flags", "Custom Fields", "Data Dictionary"]},
    "Compose": {"description": "Features related to the SQL editor for writing and running queries.", "keywords_and_aliases": ["Compose", "Query", "SQL Editor", "SmartSuggest"]},
    "Alation Analytics": {"description": "Features related to the Alation Analytics module (AAv2) and reporting on catalog usage.", "keywords_and_aliases": ["AAv2", "Alation Analytics", "Reporting", "Dashboard"]},
    "Search": {"description": "Features related to finding assets and information within the data catalog.", "keywords_and_aliases": ["Search", "Filter", "Discoverability"]},
    "Open Connector Framework": {"description": "Updates and new features for the OCF, including new connectors and enhancements to existing ones.", "keywords_and_aliases": ["OCF", "Connector", "Metadata Extraction", "MDE", "QLI"]},
    "Lineage": {"description": "Features related to viewing and managing data lineage.", "keywords_and_aliases": ["Lineage", "Impact Analysis", "Data Flow"]},
    "Security": {"description": "Features related to user authentication, authorization, and platform security.", "keywords_and_aliases": ["SCIM", "SAML", "SSO", "Sailpoint", "Okta", "Permissions", "Authentication"]},
    "Admin Settings": {"description": "Features and settings found in the Admin section of Alation.", "keywords_and_aliases": ["Admin", "Configuration", "Server Settings", "User Management"]},
    "Platform": {"description": "Core infrastructure, performance, and backend updates.", "keywords_and_aliases": ["Platform", "Performance", "Infrastructure", "Upgrade"]},
    "Public APIs": {"description": "Updates to Alation's public APIs for developers.", "keywords_and_aliases": ["API", "Endpoint", "GraphQL", "REST"]},
    "Other": {"description": "Items that do not fit into the other categories.", "keywords_and_aliases": []}
  },
  "writing_style_guide": {
    "professional_tone_rule": "Adopt a neutral, professional tone (Microsoft Style Guide). DO NOT use phrases like 'We are excited to announce...'. State facts directly.",
    "terminology_rules": { "Neo": "New User Experience", "New UI": "New User Experience", "Classic UI": "Classic User Experience", "old UI": "Classic User Experience", "DocHubs": "Document Hubs", "AAv2": "Alation Analytics" },
    "category_specific_rules": {"Public APIs": "You MUST identify the specific API endpoints, methods (e.g., GET, POST), or parameters that were changed and include these technical details in the description."},
    "feature_enhancement_writing": { "instruction": "Create a short, bolded title from the 'Summary'. On the next line, write a clear, benefit-oriented paragraph. At the end, add the Jira key in parentheses." },
    "bug_fix_writing": { "instruction": "Generate a single sentence for a bullet point starting with 'Fixed an issue where...'. At the end, add the Jira key in parentheses." }
  },
  "cloud_native_identifier": { "suffix_to_add": "(Alation Cloud Service)" }
}

# --- Helper functions ---
def build_classifier_prompt(note, type):
    if type == 'publicity':
        return f"""Analyze the ticket based on its **ultimate outcome** for the end-user. Backend work is PUBLIC if the result is a new capability, a noticeable performance improvement, or a fixed bug.
        Ticket Data: {{ "Summary": "{note.get("Summary", "")}", "Issue Type": "{note.get("Issue Type", "")}" }}
        Is this change PUBLIC or INTERNAL? Respond with a single word."""
    elif type == 'deployment':
        return f"""Analyze the engineering note to determine its deployment model.
        - **Cloud Only**: Mentions 'Alation Cloud Service', 'ACS', or is a feature that cannot exist on-prem.
        - **On-Premise Only**: Mentions 'server installation', 'update pack', 'RPM'.
        - **Both**: If it mentions both, or is a general product feature, classify it as 'Both'.
        Ticket Data: Summary: {note.get("Summary", "")}
        Your response must be one of: Cloud Only, On-Premise Only, or Both."""

def build_categorizer_prompt(note, categories_kb):
    return f"""You are an expert Alation Product Manager. Your task is to categorize an engineering ticket into ONE of the official product areas defined below.

    **Product Categories and Keywords:**
    {json.dumps(categories_kb, indent=2)}

    **Ticket to Categorize:**
    - Summary: {note.get("Summary", "")}
    - Description: {(note.get("Description", "") or "")[:300]}

    **Instructions (Chain of Thought):**
    1.  **Analyze Keywords:** Scan the summary and description for keywords that match the categories.
    2.  **Analyze Intent:** Determine the core purpose of the change.
    3.  **Justify:** Write a one-sentence justification for your choice.
    4.  **Conclude:** State the final category name. If no category is a good fit, conclude with "Other".

    Respond in a JSON format with your justification and the final category. Example:
    {{
      "justification": "The summary mentions SCIM and Sailpoint, which are related to security and authentication.",
      "category": "Security"
    }}
    """

def build_release_prompt(kb, note, category, deployment_type):
    style = kb['writing_style_guide']
    issue_type = note.get("Issue Type", "Feature").lower()

    if "bug" in issue_type or "escalation" in issue_type:
        task_instruction = f"**Task:** Write a single sentence for a Markdown bullet point. {style['bug_fix_writing']['instruction']}"
    else:
        task_instruction = f"**Task:** Write the release note. {style['feature_enhancement_writing']['instruction']}"
    
    if category in style['category_specific_rules']:
        task_instruction += f"\n**Special Instruction for this Category:** {style['category_specific_rules'][category]}"

    cloud_suffix = kb['cloud_native_identifier']['suffix_to_add']
    cloud_instruction = f"If the deployment type is 'Cloud Only', you MUST add the suffix '{cloud_suffix}' at the very end of the note, after the Jira key."

    prompt = f"""
    You are a Principal Technical Writer at {kb['company_name']}. Convert the raw engineering note for the '{category}' category into a formal release note.

    **CRITICAL WRITING RULES:**
    1.  **Professional Tone:** {style['professional_tone_rule']}
    2.  **Terminology Substitution:** Replace these internal names: {json.dumps(style['terminology_rules'])}.
    3.  **Sanitize Output:** Remove all internal jargon.
    4.  **Append Jira Key:** At the end of the note, append the 'Key', enclosed in parentheses.
    5.  **Cloud Suffix:** {cloud_instruction}

    **Raw Engineering Note:** ```json\n{json.dumps(note, indent=2)}\n```
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'final_report' not in st.session_state: st.session_state.final_report = None

with st.expander("⚙️ **Configuration**", expanded=True):
    st.info("Please provide your API key and the release details below.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password", label_visibility="collapsed", placeholder="Enter your OpenAI API Key")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        release_version = st.text_input("Release Version", "2025.3.1")
    with col2:
        build_number = st.text_input("Build Number", "2409")
    with col3:
        release_date = st.text_input("Release Date", "September 28, 2025")

# --- CORRECTED: Using st.container with border for a robust layout ---
with st.container(border=True):
    st.header("Step 1: Upload Your Content Files")
    col1, col2, col3, col4 = st.columns(4)
    with col1: epics_csv = st.file_uploader("1. Epics", type="csv")
    with col2: stories_csv = st.file_uploader("2. Stories", type="csv")
    with col3: bugs_csv = st.file_uploader("3. Bug Fixes", type="csv")
    with col4: escalations_csv = st.file_uploader("4. Support Escalations", type="csv")

    if epics_csv and stories_csv and bugs_csv and escalations_csv:
        if st.button("1️⃣ Triage & Categorize Items", type="primary", use_container_width=True):
            st.session_state.processed_data = None
            st.session_state.final_report = None
            if not api_key: st.error("Please enter your OpenAI API key in the Configuration section.")
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
                        
                        publicity_prompt = build_classifier_prompt(eng_note, 'publicity')
                        try:
                            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": publicity_prompt}], max_tokens=5, temperature=0)
                            if "PUBLIC" in response.choices[0].message.content.strip().upper():
                                deployment_prompt = build_classifier_prompt(eng_note, 'deployment')
                                dep_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": deployment_prompt}], max_tokens=10, temperature=0)
                                eng_note['Deployment'] = dep_response.choices[0].message.content.strip()

                                categorizer_prompt = build_categorizer_prompt(eng_note, KNOWLEDGE_BASE['product_categories'])
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

if st.session_state.processed_data is not None:
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
                    
                    writer_prompt = build_release_prompt(KNOWLEDGE_BASE, eng_note, eng_note['Category'], eng_note['Deployment'])
                    try:
                        writer_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                        suggestion = writer_response.choices[0].message.content.strip()
                        issue_type = eng_note.get("Issue Type", "Feature").lower()
                        
                        category = eng_note['Category']
                        if "bug" in issue_type or "escalation" in issue_type:
                            final_category = "Other Fixes" if category == "Other" else category
                            bugs_by_category[final_category].append(suggestion)
                        else:
                            if category == "Other":
                                try:
                                    final_category = suggestion.split('\n')[0].replace('**', '').strip()
                                except IndexError:
                                    final_category = eng_note.get('Summary', 'Uncategorized Feature')
                            else:
                                final_category = category
                            features_by_category[final_category].append(suggestion)
                    except Exception as e:
                        st.warning(f"Could not write note for {row.get('Summary')}: {e}")

                # Document Assembly
                main_title = f"# Release {release_version} (Build {build_number})"
                date_subtitle = f"_{release_date}_"
                report_parts = [main_title, date_subtitle]
                
                if features_by_category:
                    report_parts.append(f"\n\n**New Features and Enhancements**\n")
                    for category_key in KNOWLEDGE_BASE['product_categories']:
                        if category_key in features_by_category:
                            report_parts.append(f"\n### {category_key}\n")
                            report_parts.append("\n\n".join(features_by_category.pop(category_key)))
                    for category_key, notes in features_by_category.items():
                        report_parts.append(f"\n### {category_key}\n")
                        report_parts.append("\n\n".join(notes))

                if bugs_by_category:
                    report_parts.append(f"\n\n**Bug Fixes**\n")
                    for category_key in KNOWLEDGE_BASE['product_categories']:
                         if category_key in bugs_by_category:
                            report_parts.append(f"\n### {category_key}\n")
                            report_parts.append("\n".join(bugs_by_category.pop(category_key)))
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
