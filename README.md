# Hugging Face Chatbot (Streamlit + CLI)

Minimal chatbot using Hugging Face `transformers` that runs on CPU. Includes a Streamlit web UI and an optional CLI for quick testing.

## Features
- Streamlit chat UI (`app.py`)
- Optional CLI chatbot (`chatbot.py`)
- Configurable model via `MODEL_NAME` env var (defaults to `distilgpt2`)

## Setup
1. Create and activate a virtual environment (optional but recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Verify Streamlit works:

```bash
streamlit hello
```

## Run locally

Web app:
```bash
streamlit run app.py
```

CLI:
```bash
python chatbot.py
```

To choose a different model (small models recommended for CPU):
```bash
set MODEL_NAME=distilgpt2   # Windows PowerShell: $Env:MODEL_NAME = "distilgpt2"
```

Then run Streamlit or CLI as above.

## Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Create a new Streamlit Cloud app pointing at `app.py`
3. Ensure `requirements.txt` contains:

```
torch
transformers
streamlit
```

Optionally set the environment variable `MODEL_NAME` in the app settings.
