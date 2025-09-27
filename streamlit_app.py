import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Interactive Release Notes Assistant 🚀", layout="wide")

# --- Embedded Knowledge Base (Style Guide) ---
RELEASE_KNOWLEDGE_BASE = {
  "release_structure": { "section_order": [ "New Features", "Enhancements", "Bug Fixes" ] },
  "writing_style_guide": {
    "feature_enhancement_writing": {
      "instruction": "First, create a short, descriptive title for the update from the 'Summary'. Do not include the Jira key in the title itself. The title should be bolded. On the next line, write a clear, benefit-oriented paragraph describing the update. At the very end of the paragraph, add the Jira key in parentheses.",
      "example_format": "**This is a Clean Feature Title**\\n\\nA descriptive paragraph about the feature, explaining its value to the user. (JIRA-KEY-123)"
    },
    "bug_fix_writing": {
      "instruction": "Generate a single, concise sentence for a bullet point. Start with 'Fixed an issue where...'. At the very end of the sentence, add the Jira key in parentheses.",
      "format": "* Fixed an issue where [description of the problem]. ([JIRA-KEY-123])"
    }
  }
}

# --- Helper functions ---
def find_column_by_substring(df, substring):
    substring = substring.lower()
    for col in df.columns:
        if substring in col.lower():
            return col
    return None

def build_classifier_prompt(engineering_note):
    triage_data = { "Summary": engineering_note.get("Summary", ""), "Issue Type": engineering_note.get("Issue Type", "") }
    return f"""
    You are a discerning Principal Release Manager. Your goal is to accurately identify customer-facing changes based on their ultimate outcome for the end-user. Backend work is PUBLIC if the result is a new capability, a noticeable performance improvement, or a fixed bug.
    - PRIORITIZE THE SUMMARY: If the summary implies a new capability, lean towards PUBLIC.
    - ANALYZE THE OUTCOME: A task like "Refactor Search Indexing" is PUBLIC if it makes search faster.
    Ticket Data: {json.dumps(triage_data)}
    Is this change PUBLIC or INTERNAL? Respond with a single word.
    """

def build_release_prompt(knowledge_base, engineering_note):
    style_guide = knowledge_base['writing_style_guide']
    issue_type = engineering_note.get("Issue Type", "Feature").lower()
    if "bug" in issue_type or "escalation" in issue_type:
        task_instruction = f"**Task:** Write a single sentence for a Markdown bullet point using this format:\n`{style_guide['bug_fix_writing']['format']}`"
    else:
        task_instruction = f"**Task:** Write the release note following this instruction:\n\"{style_guide['feature_enhancement_writing']['instruction']}\""
    
    prompt = f"""
    You are a Principal Technical Writer at Alation, following the Microsoft Style Guide for professional and clear communication. Your task is to convert a raw engineering note into a formal, customer-facing release note.

    ---
    **CRITICAL WRITING RULES:**

    1.  **Professional Tone:** You MUST adopt a neutral and professional tone. **DO NOT** use overly enthusiastic or marketing-focused phrases like "We are excited to announce," "We are pleased to introduce," or "We're happy to share." State the facts directly.
    2.  **Terminology Substitution:** You MUST replace internal codenames with their official public-facing terms.
        - If you see "Neo", "New UI", or "Neo UI", replace it with **"New User Experience"**.
        - If you see "Classic UI" or "old UI", replace it with **"Classic User Experience"**.
    3.  **Sanitize Output:** Remove all internal jargon (e.g., 'pid 1', 'master branch').
    4.  **Append Jira Key:** At the end of the note, you MUST append the 'Key' (the Jira key), enclosed in parentheses. Example: `(AL-12345)`.
    ---

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Interactive Release Notes Assistant 🚀")

if 'final_report' not in st.session_state: st.session_state.final_report = None
if 'summary_data' not in st.session_state: st.session_state.summary_data = None

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The style guide is embedded in the application.")
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
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        else:
            client = openai.OpenAI(api_key=api_key)
            all_dfs = {
                "Epics": pd.read_csv(epics_csv).fillna(''), "Stories": pd.read_csv(stories_csv).fillna(''),
                "Bugs": pd.read_csv(bugs_csv).fillna(''), "Escalations": pd.read_csv(escalations_csv).fillna('')
            }
            progress_bar = st.progress(0)
            total_rows = sum(len(df) for df in all_dfs.values())
            processed_rows = 0
            public_items_raw = []

            for name, df in all_dfs.items():
                for index, row in df.iterrows():
                    processed_rows += 1
                    progress_bar.progress(processed_rows / total_rows, text=f"Triaging {name}: {row.get('Summary', '')[:30]}...")
                    eng_note = row.to_dict()
                    classifier_prompt = build_classifier_prompt(eng_note)
                    try:
                        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": classifier_prompt}], max_tokens=5, temperature=0)
                        classification = response.choices[0].message.content.strip().upper()
                        if "PUBLIC" in classification:
                            public_items_raw.append(eng_note)
                    except Exception as e:
                        st.warning(f"Could not classify {eng_note.get('Summary')}: {e}")
            
            df_public = pd.DataFrame(public_items_raw).fillna('')
            df_public['Include'] = True
            
            public_epic_keys = set(df_public[df_public['Issue Type'] == 'Epic']['Key'])
            
            def should_exclude(row):
                if row['Issue Type'] == 'Story' and row['parent'] in public_epic_keys:
                    return False
                return True

            df_public['Include'] = df_public.apply(should_exclude, axis=1)
            st.session_state.classified_data = df_public
            st.success(f"Triage complete. Found {len(df_public)} potentially public items for your review.")

if st.session_state.classified_data is not None:
    st.header("Step 2: Review and Approve Items")
    st.warning("Uncheck any items you want to exclude from the final document.")
    
    edited_df = st.data_editor(
        st.session_state.classified_data,
        column_config={"Include": st.column_config.CheckboxColumn("Include?", default=True)},
        disabled=["Key", "Summary", "Issue Type", "parent", "Description"],
        height=400,
        use_container_width=True
    )
    
    approved_df = edited_df[edited_df['Include']]
    st.info(f"You have selected **{len(approved_df)}** items to include in the release notes.")

    if st.button("2️⃣ Generate Document for Approved Items"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        else:
            final_results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
            client = openai.OpenAI(api_key=api_key)
            
            progress_bar = st.progress(0, text="Writing notes...")
            total_to_write = len(approved_df)
            
            for index, row in approved_df.iterrows():
                progress_bar.progress((index + 1) / total_to_write, text=f"Writing: {row.get('Summary', '')[:30]}...")
                eng_note = row.to_dict()
                writer_prompt = build_release_prompt(RELEASE_KNOWLEDGE_BASE, eng_note)
                try:
                    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
                    suggestion = response.choices[0].message.content.strip()
                    issue_type = eng_note.get("Issue Type", "Feature").lower()
                    if "bug" in issue_type or "escalation" in issue_type: final_results["Bug Fixes"].append(suggestion)
                    elif "enhancement" in issue_type: final_results["Enhancements"].append(suggestion)
                    else: final_results["New Features"].append(suggestion)
                except Exception as e:
                    st.warning(f"Could not write note for {row.get('Summary')}: {e}")

            month_year = datetime.now().strftime('%B %Y')
            report_parts = [f"# Release {release_version}", f"_{month_year}_"]
            for section in RELEASE_KNOWLEDGE_BASE['release_structure']['section_order']:
                if final_results.get(section):
                    report_parts.append(f"\n\n**{section}**\n")
                    if section == "Bug Fixes": report_parts.append("\n".join(final_results[section]))
                    else: report_parts.append("\n\n".join(final_results[section]))
            
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
