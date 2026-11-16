# 💊 Multi-Agent Pharmacy Assistant (RAG & LLM Workflow)

An intelligent, multi-agent Pharmacy Assistant engineered to streamline medication information retrieval, symptom-based drug recommendations, clinical data processing, and context-aware patient consultations through automated workflow orchestration.

---

## 📋 Overview

Navigating complex pharmaceutical data, dosages , pricing , and inventory availability requires both high accuracy and contextual understanding. This project combines a **Multi-Agent Architecture (LangGraph)** with a **Retrieval-Augmented Generation (RAG)** pipeline and structured clinical rule matching to retrieve verified drug data from local medical guides and inventory sheets, query external **openFDA** drug labels, summarize clinical insights via a local Large Language Model (**TinyLlama**), and maintain multi-turn conversation state using **Buffer Memory** .

---

## ✨ Key Features

* **🤖 Multi-Agent Workflow Orchestration:** Coordinates three specialized agents (`Data Retrieval Agent` → `Processing & Summarization Agent` → `Automated Action Agent`) using a **LangGraph** state machine  (`StateGraph`).
* **🔍 Hybrid RAG & Structured Retrieval Pipeline:**
  * **Vector Search & Indexing:** Uses `all-MiniLM-L6-v2` HuggingFace embeddings and **FAISS** vector indexing over a 120-drug medical reference PDF and pharmacy inventory CSV.
  * **External API Integration:** Connects to the **openFDA Drug Label API** for real-time safety warnings when requested.
  * **Symptom & Pediatric Safety Engine:** Maps user-described symptoms to appropriate OTC medications while enforcing strict pediatric safety filters and red-flag emergency detection.
* **🧠 Grounded LLM Data Processing & Summarization:** Uses **TinyLlama-1.1B-Chat** with anti-hallucination grounding verification and deterministic extractive fallbacks to generate accurate , concise answers.
* **💬 Context-Aware Conversational Buffer Memory:** Implements a persistent `ChatSession` memory state that tracks conversation history (`last_question`, `last_answer`) as well as previously displayed drug lists (`pool`, `shown`), enabling natural follow-ups such as *"What is the price of it?"*, *"Get the cheapest"*, or *"Summarize"*.
* **📄 Automated Reporting & Interactive Web UI:** Includes a **Gradio** chat interface and an automated Action Agent that exports consultation summaries to `automated_report.md`.

---

## 🏗️ System Architecture

```text
User Query + Session Buffer Memory (History & Previous Drug Pool)
                               │
                               ▼
              ┌─────────────────────────────────┐
              │   LangGraph Workflow Pipeline   │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │   Agent 1: Data Retrieval       │
              │  • Intent & Symptom Detection   │
              │  • FAISS Vector Store (PDF/CSV) │
              │  • openFDA External API         │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │   Agent 2: Processing & Summary │
              │  • TinyLlama-1.1B-Chat Inference│
              │  • Grounding & Hallucination    │
              │    Validation + Fallback        │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │   Agent 3: Automated Action     │
              │  • Generates automated_report.md│
              └────────────────┬────────────────┘
                               │
                               ▼
            Gradio Chat UI Response + Memory Update


## 🛠️ Tech Stack
Language: Python 3.10+

Agent Orchestration: LangGraph (StateGraph), LangChain (0.2.16)

LLM & Embeddings: HuggingFace Transformers (TinyLlama/TinyLlama-1.1B-Chat-v1.0), sentence-transformers/all-MiniLM-L6-v2

Vector Database & Indexing: FAISS (faiss-cpu)

External Data & APIs: PyPDF, Pandas, openFDA REST API

User Interface: Gradio


📁 Project Structure
multi-agent-pharmacy-assistant/
├── data/
│   ├── Comprehensive  Medical & Pharmacy Guide (120 Drugs).pdf
│   └── Pharmacy_Inventory_120_New_Drugs.csv
├── src/
│   ├── __init__.py
│   └── backend.py          # RAG pipeline, LangGraph agents, symptom matcher & ChatSession memory
├── app.py                  # Gradio interactive web UI with session state
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored cache and generated report files
└── README.md               # Project documentation


🚀 Installation & Usage
Option 1: Run Locally
   1. Clone the Repository:
             git clone [(https://github.com/Waheeed09/RAG-Based-Intelligent-Pharmaceutical-Information-System-.git)]https://github.com/Waheeed09/RAG-Based-Intelligent-Pharmaceutical-Information-System-.git
cd multi-agent-pharmacy-assistant

  2.  Create and Activate a Virtual Environment (Recommended):
             python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

   3.  Install Dependencies:
              pip install -r requirements.txt
   4.  Launch the Application:
              python app.py
   Open the local URL displayed in the terminal (typically http://127.0.0.1:7860) in your browser.

Option 2: Run on Google Colab (GPU Accelerated)
You can run the entire project directly on Google Colab in a single cell without local setup:
     %cd /content/RAG-Based-Intelligent-Pharmaceutical-Information-System-/multi-agent-pharmacy-assistant
!pip install -q -r requirements.txt
!python app.py


💬 Example Conversation Flow

User: What is the standard dosage of Zyrtec?
Assistant: Zyrtec 10mg Tabs (Cetirizine 10mg): 1 tablet once daily, preferably at bedtime.

User: I have a cold, recommend me a medicine
Assistant: Get well soon! For (cold / flu), these are the best matches in the medical guide and the pharmacy inventory:
1. Congestal Tabs (Paracetamol + Pseudoephedrine + Chlorpheniramine) - 50 EGP | Dosage: 1 tablet every 8 hrs after meals...
2. Panadol Cold + Flu Day (Paracetamol + Phenylephrine) - 76 EGP...
3. 123 Cold (Paracetamol + Pseudoephedrine + Chlorpheniramine) - 42 EGP...

User: get the cheapest
Assistant: [Context retained via Session Buffer Memory]
The cheapest option from the list above is:
1. 123 Cold (Paracetamol + Pseudoephedrine + Chlorpheniramine) - 42 EGP...

User: summarize
Assistant: Summary - top 2 options:
1. Congestal Tabs (Paracetamol + Pseudoephedrine + Chlorpheniramine) - 50 EGP
2. Panadol Cold + Flu Day (Paracetamol + Phenylephrine) - 76 EGP
