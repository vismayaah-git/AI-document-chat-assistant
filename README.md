# AI-Powered Document & Chat Assistant

A Streamlit application that works as an AI-powered document assistant when a text-based PDF is uploaded, and as a general chat assistant when no PDF is uploaded. For document questions, it combines semantic search, keyword search, and reranking to find relevant passages.

## Features

- Upload a PDF and ask questions about its contents.
- Combine FAISS vector search with BM25 keyword search.
- Rerank retrieved passages with a CrossEncoder model.
- Use the chatbot without uploading a PDF, or remove the uploaded PDF to switch from PDF Q&A to general chat.
- Reuse each browser session's PDF index across Streamlit reruns.
- Handle scanned PDFs and common Groq API errors with clear messages.

## Requirements

- Python 3.12
- A Groq API key and access to the model set in `GROQ_MODEL`
- Internet access to download the embedding and reranker models on 
first use

## Setup

Create and activate a virtual environment from the project directory.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies and create your local environment file:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `GROQ_API_KEY` to your own key. The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL` to another active model if needed.

Do not commit or share `.env`. It contains your private API key.

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests mock Groq requests, so they do not require an API key or make API calls.

## How it works

1. `pdf_processor.py` extracts selectable text from the PDF and splits it into overlapping chunks.
2. `embeddings.py` creates vector embeddings.
3. `retriever.py` combines FAISS and BM25 results, then reranks candidates.
4. `llm.py` sends the question and retrieved context to Groq.
5. `app.py` provides the Streamlit interface and manages the PDF index for the current browser session.

Scanned PDFs need OCR before they can be searched. Uploaded PDFs and their search indexes are not saved to disk by the app.

When no PDF is uploaded, questions are sent to Groq without PDF context, and the app works as a general chatbot.
