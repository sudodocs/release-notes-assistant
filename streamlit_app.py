import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Release Notes Assistant 🚀",
    page_icon="📝",
    layout="wide"
)

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
def build_classifier_prompt(engineering_note):
    """Builds a more nuanced prompt to classify a note as PUBLIC or INTERNAL."""
    triage_data = { "Summary": engineering_note.get("Summary", ""), "Issue Type": engineering_note.get("Issue Type", "") }
    return f"""
    You are a discerning Principal Release Manager. Your goal is to accurately identify customer-facing changes.
    Base your decision on the **ultimate outcome** for the end-user. Backend work is PUBLIC if the result is a new capability, a noticeable performance improvement, or a fixed bug.
    - PRIORITIZE THE SUMMARY: If the summary implies a new capability, lean towards PUBLIC.
    - ANALYZE THE OUTCOME: A task like "Refactor Search Indexing" is PUBLIC if it makes search faster. A task like "Update Frontend Dependencies" is PUBLIC if it patches a security vulnerability.
    - DO NOT over-rely on keywords like 'refactor' or 'pipeline'.
    Ticket Data: {json.dumps(triage_data)}
    Is this change PUBLIC or INTERNAL? Respond with a single word.
    """

def build_benefit_summary_prompt(engineering_note):
    """Builds a prompt to summarize the customer-facing benefit."""
    return f"""
    You are a Product Manager. Analyze the following engineering ticket.
    Your task is to write a single, concise sentence that describes the primary **customer-facing benefit** of this change.
    If there is no clear, direct customer benefit, respond with the exact phrase "No clear benefit."

    **Ticket Data:**
    Summary: {engineering_note.get("Summary", "")}
    Description: {(engineering_note.get("Description", "") or "")[:400]}

    What is the primary customer-facing benefit?
    """

def build_release_prompt(knowledge_base, engineering_note, benefit_summary):
    """Builds the final prompt to write the release note, using the benefit summary as context."""
    style_guide = knowledge_base['writing_style_guide']
    issue_type = engineering_note.get("Issue Type", "Feature").lower()
    
    if "bug" in issue_type or "escalation" in issue_type:
        task_instruction = f"**Task:** Using the provided benefit summary as context, write a single sentence for a Markdown bullet point using this exact format:\n`{style_guide['bug_fix_writing']['format']}`"
    else:
        task_instruction = f"**Task:** Using the provided benefit summary as context, write the release note following this instruction:\n\"{style_guide['feature_enhancement_writing']['instruction']}\""
    
    prompt = f"""
    You are a Principal Technical Writer at Alation. You have been given an engineering note and a Product Manager's summary of the customer benefit.
    Your task is to convert the raw engineering note into a formal, customer-facing release note.
    **CRITICAL RULE:** Remove all internal jargon (e.g., 'pid 1', 'master branch').
    **Crucial Instruction:** At the end of the note, you MUST append the 'Key' (the Jira key), enclosed in parentheses.

    **Product Manager's Benefit Summary:**
    {benefit_summary}

    **Raw Engineering Note:**
    ```json
    {json.dumps(engineering_note, indent=2)}
    ```
    
    {task_instruction}
    """
    return prompt

# --- Main Application Logic ---
st.title("Intelligent Release Notes Assistant 🚀")

if 'final_report' not in st.session_state: st.session_state.final_report = None
if 'summary_data' not in st.session_state: st.session_state.summary_data = None

with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("The release notes style guide is embedded in the application.")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    release_version = st.text_input("Enter Release Version (e.g., 2025.3.1)", "2025.3.1")

st.header("Step 1: Upload Your Content Files")
col1, col2, col3, col4 = st.columns(4)
with col1: epics_csv = st.file_uploader("1. Epics", type="csv")
with col2: stories_csv = st.file_uploader("2. Stories", type="csv")
with col3: bugs_csv = st.file_uploader("3. Bug Fixes", type="csv")
with col4: escalations_csv = st.file_uploader("4. Support Escalations", type="csv")

st.header("Step 2: Generate Notes")
if st.button("📝 Generate Release Notes Document"):
    st.session_state.final_report = None
    st.session_state.summary_data = None
    
    if not all([epics_csv, stories_csv, bugs_csv, escalations_csv, api_key]):
        st.error("Please upload all four CSV files and enter an API key to proceed.")
    else:
        # --- Data Loading and AI Chain ---
        df_epics = pd.read_csv(epics_csv).fillna('')
        df_stories = pd.read_csv(stories_csv).fillna('')
        df_bugs = pd.read_csv(bugs_csv).fillna('')
        df_escalations = pd.read_csv(escalations_csv).fillna('')
        all_dfs = {"Epics": df_epics, "Stories": df_stories, "Bugs": df_bugs, "Escalations": df_escalations}
        
        client = openai.OpenAI(api_key=api_key)
        public_items = {"Epics": [], "Stories": [], "Bugs": []}
        skipped_items = []
        progress_bar = st.progress(0)
        total_rows = sum(len(df) for df in all_dfs.values())
        processed_rows = 0

        # --- Step 1: AI Triage ---
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
                        if name in ["Bugs", "Escalations"]: public_items["Bugs"].append(eng_note)
                        else: public_items[name].append(eng_note)
                    else: skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), "Classified as Internal"))
                except Exception as e: skipped_items.append((eng_note.get('Key'), eng_note.get('Summary'), f"Triage failed: {e}"))
        
        # --- De-duplication Logic ---
        public_epic_keys = {note['Key'] for note in public_items["Epics"]}
        # Filter stories whose parent is a public epic
        stories_to_keep = [story for story in public_items["Stories"] if story.get('parent') not in public_epic_keys]
        stories_skipped_count = len(public_items["Stories"]) - len(stories_to_keep)
        if stories_skipped_count > 0:
            skipped_items.extend([(story.get('Key'), story.get('Summary'), "Skipped (Parent Epic is public)") for story in public_items["Stories"] if story.get('parent') in public_epic_keys])
        
        # Filter bugs whose parent is a new public feature
        public_feature_keys = public_epic_keys.union({story['Key'] for story in stories_to_keep})
        bugs_to_keep = [bug for bug in public_items["Bugs"] if bug.get('parent') not in public_feature_keys]
        bugs_skipped_count = len(public_items["Bugs"]) - len(bugs_to_keep)
        if bugs_skipped_count > 0:
             skipped_items.extend([(bug.get('Key'), bug.get('Summary'), "Skipped (Bug for new feature)") for bug in public_items["Bugs"] if bug.get('parent') in public_feature_keys])

        features_to_process = public_items["Epics"] + stories_to_keep
        bugs_to_process = bugs_to_keep
        
        # --- Step 2 & 3: AI Summarizer and Writer ---
        final_results = {"New Features": [], "Enhancements": [], "Bug Fixes": []}
        total_to_write = len(features_to_process) + len(bugs_to_process)
        written_count = 0
        
        for note in features_to_process + bugs_to_process:
            written_count += 1
            progress_bar.progress(written_count / total_to_write, text=f"Writing note for {note.get('Summary', '')[:30]}...")
            # Step 2: Summarize Benefit
            benefit_prompt = build_benefit_summary_prompt(note)
            benefit_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": benefit_prompt}], max_tokens=60, temperature=0)
            benefit_summary = benefit_response.choices[0].message.content.strip()

            if "No clear benefit" in benefit_summary:
                skipped_items.append((note.get('Key'), note.get('Summary'), "No clear customer benefit found"))
                continue

            # Step 3: Write Final Note
            writer_prompt = build_release_prompt(RELEASE_KNOWLEDGE_BASE, note, benefit_summary)
            writer_response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": writer_prompt}])
            suggestion = writer_response.choices[0].message.content.strip()

            issue_type = note.get("Issue Type", "Feature").lower()
            if "bug" in issue_type or "escalation" in issue_type: final_results["Bug Fixes"].append(suggestion)
            else: final_results["New Features"].append(suggestion)

        # --- Assemble Final Document ---
        progress_bar.progress(1.0, text="Assembling final document...")
        month_year = datetime.now().strftime('%B %Y')
        report_parts = [f"# Release {release_version}", f"_{month_year}_"]
        for section in RELEASE_KNOWLEDGE_BASE['release_structure']['section_order']:
            if final_results.get(section):
                report_parts.append(f"\n\n**{section}**\n")
                if section == "Bug Fixes": report_parts.append("\n".join(final_results[section]))
                else: report_parts.append("\n\n".join(final_results[section]))
        
        st.session_state.final_report = "\n".join(report_parts)
        st.session_state.summary_data = {
            "total": total_rows, "processed_count": len(features_to_process) + len(bugs_to_process),
            "skipped_count": len(skipped_items), "processed_list": [(note.get('Key'), note.get('Summary')) for note in features_to_process + bugs_to_process],
            "skipped_list": skipped_items
        }
        st.success("✅ Release notes document generated successfully!")

if st.session_state.summary_data:
    summary = st.session_state.summary_data
    st.info(f"**Processing Summary:** {summary['processed_count']} notes included, {summary['skipped_count']} notes skipped.")
    with st.expander("Show Detailed Processing Report"):
        st.markdown("#### ✅ Notes Included in Document")
        for jira_key, summary_text in summary['processed_list']: st.text(f"- {jira_key}: {summary_text}")
        st.markdown("#### ⏩ Notes Skipped")
        for jira_key, summary_text, reason in summary['skipped_list']: st.text(f"- {jira_key}: {summary_text} (Reason: {reason})")

if st.session_state.final_report:
    st.header("Step 3: Download Report")
    st.markdown("### Preview")
    st.markdown(st.session_state.final_report, unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Release Notes (.md)",
        data=st.session_state.final_report.encode('utf-8'),
        file_name=f"Release_Notes_{release_version}.md",
        mime="text/markdown",
    )
