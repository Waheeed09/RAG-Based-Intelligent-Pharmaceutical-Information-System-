import glob
import os
import gradio as gr
from src import backend as nbk  # أو import backend as nbk لو في نفس الفولدر


def resolve_file(env_var: str, pattern: str) -> str:
    """Find file via env var, ./data/, or current directory."""
    if os.getenv(env_var) and os.path.exists(os.getenv(env_var)):
        return os.getenv(env_var)
    for folder in ["data", ".", "/content"]:
        matches = glob.glob(os.path.join(folder, pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find file matching '{pattern}'. Place it in 'data/'.")


PDF_PATH = resolve_file("PDF_PATH", "*.pdf")
CSV_PATH = resolve_file("CSV_PATH", "*.csv")

print(f"Using PDF: {PDF_PATH}")
print(f"Using CSV: {CSV_PATH}")

print("Building vector store & multi-agent workflow...")
VECTOR_STORE = nbk.load_and_process_documents(PDF_PATH, CSV_PATH)
WORKFLOW = nbk.build_workflow()
print("System Ready!")


def create_session():
    return nbk.ChatSession(WORKFLOW, VECTOR_STORE)


def chat(message, history, session):
    message = (message or "").strip()
    if not message:
        return "Please type a question first.", session

    if session is None:
        session = create_session()

    try:
        reply = session.ask(message)
        return reply.strip() or nbk.NO_INFO, session
    except Exception as e:
        print(f"Workflow error: {type(e).__name__}: {e}")
        return "Sorry, something went wrong while answering. Please try again.", session


EXAMPLES = [
    ["What is the standard dosage of Zyrtec?"],
    ["I have a cold, recommend me a medicine"],
    ["get the cheapest"],
    ["summarize"],
]

with gr.Blocks(title="Pharmacy Assistant", theme=gr.themes.Soft()) as demo:
    session_state = gr.State(create_session)
    gr.Markdown("# 💊 Multi-Agent Pharmacy Assistant\nAsk about drug dosage, warnings, recommendations, price, or stock.")

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Type your symptom or drug question here...", label="Your Message")
    clear = gr.Button("Clear Conversation")

    def respond(user_msg, chat_history, session):
        bot_reply, session = chat(user_msg, chat_history, session)
        chat_history = chat_history + [(user_msg, bot_reply)]
        return "", chat_history, session

    def reset_chat():
        return [], create_session()

    msg.submit(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])
    clear.click(reset_chat, None, [chatbot, session_state])
    gr.Examples(examples=EXAMPLES, inputs=msg)

if __name__ == "__main__":
    demo.launch(share=True)