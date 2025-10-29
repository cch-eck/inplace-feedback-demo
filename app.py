# app.py
import gradio as gr
from model import generate_response

def chat_fn(user_input, history):
    """history is a list of [user, bot] pairs."""
    response = generate_response(user_input)
    history.append((user_input, response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Simple Qwen Chatbot Demo")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your message")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat_fn, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860)
