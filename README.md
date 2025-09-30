# Intelligent Release Notes Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) An intelligent Streamlit assistant that automates the generation of professional, publication-ready release notes from Jira CSV exports using a customizable AI-powered workflow.

This application transforms the tedious, manual process of writing release notes into a streamlined, semi-automated workflow. By leveraging a configurable knowledge base and powerful Large Language Models (LLMs) like OpenAI, Google Gemini, and Hugging Face, it produces consistent, high-quality documentation with minimal human effort.

---
## 🚀 Use-Cases & Benefits

This tool is designed for **technical writers, product managers, and release managers** who need to create accurate and professional release notes efficiently.

* **Automate Tedious Triage:** Instead of manually sifting through hundreds of Jira tickets, the application uses AI to automatically identify which items are public-facing and relevant for a release.
* **Ensure Consistency:** By using a centralized `knowledge_base.json`, all generated content adheres to a consistent tone, style, and terminology, eliminating variations between different writers or releases.
* **Accelerate Turnaround Time:** Drastically reduce the time it takes to go from a list of completed Jira tickets to a finalized, formatted release notes document. What used to take days can now be done in minutes.
* **Improve Accuracy:** The AI is instructed to extract specific technical details (like API endpoints) and follow precise formatting rules, reducing the risk of human error.
* **Flexible & Adaptable:** The entire logic, from product categories to writing style, is controlled by a single JSON file. This makes the application easy to adapt to any company's product structure and branding without changing the core code.

---
## ✨ Key Features

* **AI-Powered Triage:** Automatically filters Jira exports to find public-facing features, enhancements, and bug fixes.
* **Intelligent Categorization:** Sorts items into product categories defined in a customizable knowledge base.
* **Automated Content Generation:** Writes clear, benefit-oriented descriptions for new features and concise summaries for bug fixes.
* **Human-in-the-Loop Review:** Provides an interactive table where a user can review, edit, and approve or reject the AI's suggestions before final document generation.
* **Multi-Format Export:** Generates the final release notes in both **Markdown (.md)** and **reStructuredText (.rst)**, complete with a table of contents and cross-references suitable for technical documentation platforms.
* **Multi-Provider LLM Support:** Flexibly integrates with **OpenAI (GPT-4o)**, **Google Gemini**, and **Hugging Face** models.
* **Knowledge Base Driven:** All logic, prompts, writing styles, and product definitions are managed in an external `knowledge_base.json` file, making the application highly customizable.

---
## Workflow: From CSV to Release Notes

The application guides the user through a simple, multi-step process:

1.  **Configuration:** The user selects their preferred AI provider (OpenAI, Gemini, Hugging Face) and provides an API key. They also input the URL for the project's `knowledge_base.json` and enter release-specific details like version number and date.

2.  **Upload Jira Exports:** The user uploads four CSV files exported from Jira: Epics, Stories, Bugs, and Support Escalations.

3.  **Triage & Categorize:** With a single click, the application processes all tickets. It uses the AI and the knowledge base to perform an initial triage, identifying public items and assigning them a deployment model (Cloud, On-prem, etc.) and a product category.

4.  **Review & Approve:** The AI's suggestions are displayed in an interactive table. The user can quickly review the suggested category and deployment model, make corrections, and uncheck any items they wish to exclude from the final document.

5.  **Generate Document:** Once the user approves the list, the application sends each item to the AI with specific instructions based on its category (feature or bug). The AI writes the final, polished descriptions.

6.  **Assemble & Download:** The application assembles the AI-written content into a complete, structured document. It generates a Markdown preview and provides download buttons for both `.md` and `.rst` files.

---
## 🔧 Setup & Installation

To run this application locally, follow these steps:

1.  **Prerequisites:**
    * Python 3.8+
    * An API key for OpenAI, Google Gemini, or Hugging Face.

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/intelligent-release-notes.git](https://github.com/your-username/intelligent-release-notes.git)
    cd intelligent-release-notes
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```

The application will open in a new browser tab.
