import os
from typing import List, Dict

import streamlit as st
from transformers import pipeline

# Default model name, can be changed via environment variable
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")

@st.cache_resource(show_spinner=False)
#Setting up a ready-to-use text generation model
def load_text_generation_pipeline(model_name: str):
    return pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=model_name,
        device=-1,  # CPU (-1 means no GPU)
    )

# Quick test to see if the model is working
gen = load_text_generation_pipeline(DEFAULT_MODEL_NAME)
print(gen("Hello there")[0]["generated_text"])


def build_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """
    Turn a list of messages into a single prompt string.

    history = [{"role": "user"|"assistant", "content": "..."}]
    """
    preamble = "Below is a conversation between a helpful AI assistant and a user.\n"
    rendered = ""
    for m in history:
        if m["role"] == "user":
            rendered += f"User: {m['content']}\n"
        else:
            rendered += f"Assistant: {m['content']}\n"

    return f"{preamble}{rendered}User: {user_message}\nAssistant:"


def extract_assistant_reply(generated_text: str) -> str:
    # Take text after the last "Assistant:" marker if present
    reply = generated_text.split("Assistant:")[-1] if "Assistant:" in generated_text else generated_text
    # Stop at the next turn marker if present
    for stop_token in ["\nUser:", "\nuser:", "\nHuman:"]:
        if stop_token in reply:
            reply = reply.split(stop_token)[0]
            break
    return reply.strip()


def generate_reply(gen, history: list[dict[str, str]], user_message: str) -> str:
    prompt = build_prompt(history, user_message)
    outputs = gen(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=gen.tokenizer.eos_token_id,
        eos_token_id=gen.tokenizer.eos_token_id,
    )
    full_text = outputs[0]["generated_text"]
    return extract_assistant_reply(full_text)


def init_session_state():
    """Initialize chat history and model in Streamlit's session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "..."}]
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL_NAME
    if "generator" not in st.session_state:
        st.session_state.generator = load_text_generation_pipeline(st.session_state.model_name)


def sidebar_ui():
    """Sidebar for model settings and chat controls."""
    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "Hugging Face model",
            value=st.session_state.model_name,
            help="Use a small model like 'distilgpt2' for quick tests."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Load model"):
                with st.spinner("Loading model… this can take a while"):
                    st.session_state.model_name = model_name
                    st.session_state.generator = load_text_generation_pipeline(model_name)
                st.success(f"Loaded: {model_name}")

        with col_b:
            if st.button("Clear chat"):
                st.session_state.messages = []

        st.markdown("---")
        st.caption("Tip: Set environment variable `MODEL_NAME` to change the default model.")


def main():
    st.set_page_config(page_title="HF Chatbot", page_icon="💬")
    init_session_state()
    sidebar_ui()

    st.title("💬 Hugging Face Chatbot")
    st.markdown("A minimal, CPU-friendly chatbot using `transformers`.")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask me anything…")
    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = generate_reply(
                        st.session_state.generator,
                        st.session_state.messages,
                        user_input
                    )
                except Exception as e:
                    reply = f"Sorry, an error occurred: {e}"
                st.markdown(reply)

        # Save bot reply
        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
