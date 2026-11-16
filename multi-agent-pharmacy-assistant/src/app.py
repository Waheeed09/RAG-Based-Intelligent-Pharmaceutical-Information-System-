"""
Chat UI (Gradio) for final_project_with_data.ipynb

Reads the notebook's code cells and runs them as-is, then sends every chat
message through the agents (Retriever -> Processor -> Action). Only the
answer is shown in the chat. Dark mode is enabled by default.

Usage:
    pip install gradio
    python app.py
Keep this file next to the notebook, the PDF and the CSV.
"""

import json
import os
import sys
import types

import gradio as gr

# Paths can be overridden with environment variables
NOTEBOOK_PATH = os.getenv("NOTEBOOK_PATH", "final_project.ipynb")
PDF_PATH = os.getenv("PDF_PATH", "Comprehensive Medical & Pharmacy Guide (120 Drugs).pdf")
CSV_PATH = os.getenv("CSV_PATH", "Pharmacy_Inventory_120_New_Drugs.csv")


def load_notebook_module(path: str) -> types.ModuleType:
    """Execute the notebook's code cells in a fresh module (Jupyter commands are skipped)."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    # Not named __main__, so the notebook's test block does not run
    module = types.ModuleType("notebook_backend")
    sys.modules["notebook_backend"] = module

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = [l for l in source.splitlines() if not l.lstrip().startswith(("!", "%"))]
        code = "\n".join(lines)
        if code.strip():
            exec(compile(code, "<notebook cell>", "exec"), module.__dict__)
    return module


print("Loading notebook code (this loads TinyLlama, may take a while)...")
nbk = load_notebook_module(NOTEBOOK_PATH)

print("Building vector store + workflow...")
VECTOR_STORE = nbk.load_and_process_documents(PDF_PATH, CSV_PATH)
WORKFLOW = nbk.build_workflow()
print("Ready.")


def last_turn(history):
    """Return (last user message, last assistant message) for any Gradio history format."""
    def text_of(content):
        if isinstance(content, list):
            return " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
        return str(content or "")

    question = answer = ""
    for h in history or []:
        if isinstance(h, dict):                      # messages format
            if h.get("role") == "user":
                question = text_of(h.get("content"))
            elif h.get("role") == "assistant":
                answer = text_of(h.get("content"))
        elif isinstance(h, (list, tuple)) and len(h) == 2:   # tuples format
            question, answer = text_of(h[0]) or question, text_of(h[1]) or answer
    return question, answer


def chat(message, history):
    message = (message or "").strip()
    if not message:
        return "Please type a question first."

    last_question, last_answer = last_turn(history)
    state = {
        "task": message,
        "vector_store": VECTOR_STORE,
        "retrieved_data": "",
        "processed_summary": "",
        "final_output_path": "",
        "log": [],
        "last_question": last_question,
        "last_answer": last_answer,
    }
    try:
        result = WORKFLOW.invoke(state)
        return result["processed_summary"].strip() or nbk.NO_INFO
    except Exception as e:
        print(f"Workflow error: {type(e).__name__}: {e}")
        return "Sorry, something went wrong while answering. Please try again."


EXAMPLES = [
    "What is the standard dosage of Zyrtec?",
    "What is the maximum daily dose of Panadol Advance?",
    "Which drug is used for acid reflux?",
]

# Gradio switches to dark mode when <body> has the "dark" class
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

with gr.Blocks(title="Pharmacy Assistant") as demo:
    gr.ChatInterface(
        fn=chat,
        title="💊 Pharmacy Assistant",
        description="Ask about drug dosage, warnings, price, or stock.",
        examples=EXAMPLES,
    )
    demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_JS)

if __name__ == "__main__":
    demo.launch(share=True)
