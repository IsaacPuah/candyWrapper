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
