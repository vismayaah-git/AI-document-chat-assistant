import hashlib
from io import BytesIO

import streamlit as st
from groq import APIConnectionError, APIStatusError, AuthenticationError, NotFoundError, RateLimitError
from pypdf.errors import PdfReadError

from pdf_processor import extract_text_from_pdf, split_text
from embeddings import create_embeddings
from retriever import create_vector_store, create_bm25, hybrid_search, rerank
from llm import GroqConfigurationError, generate_answer


def clear_pdf_session():
    for key in ("pdf_digest", "pdf_chunks", "pdf_vector_store", "pdf_bm25"):
        st.session_state.pop(key, None)


def prepare_pdf(pdf_bytes: bytes):
    text = extract_text_from_pdf(BytesIO(pdf_bytes))
    chunks = split_text(text)
    if not chunks:
        return [], None, None

    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    bm25 = create_bm25(chunks)
    return chunks, vector_store, bm25


st.title("📄 PDF QA + General Chatbot")

pdf = st.file_uploader("Upload a PDF", type="pdf")
chunks = []
vector_store = None
bm25 = None

if pdf:
    pdf_bytes = pdf.getvalue()
    # Hash the content so the index can be reused across reruns, but rebuilt for a new PDF.
    pdf_digest = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("pdf_digest") != pdf_digest:
        # Drop stale index data before processing the newly uploaded document.
        clear_pdf_session()

        try:
            with st.spinner("Reading PDF and building its search index..."):
                chunks, vector_store, bm25 = prepare_pdf(pdf_bytes)
        except PdfReadError as error:
            st.error(f"Could not read this PDF: {error}")
        else:
            st.session_state["pdf_digest"] = pdf_digest
            st.session_state["pdf_chunks"] = chunks
            st.session_state["pdf_vector_store"] = vector_store
            st.session_state["pdf_bm25"] = bm25
    else:
        # Streamlit reruns the script on interaction; reuse this session's index.
        chunks = st.session_state["pdf_chunks"]
        vector_store = st.session_state["pdf_vector_store"]
        bm25 = st.session_state["pdf_bm25"]

    if chunks:
        st.success("PDF processed successfully!")
    elif st.session_state.get("pdf_digest") == pdf_digest:
        st.warning( "No selectable text was found in this PDF. "
                    "Scanned PDFs need OCR before they can be searched." )
else:
    # Clear the previous document's index when the uploader is emptied.
    clear_pdf_session()

question = st.chat_input("Ask something...")

if question:
    if pdf and not chunks:
        st.info("Upload a PDF with selectable text, or remove the PDF to chat generally.")
    else:
        try:
            if pdf:
                vector_results, keyword_results = hybrid_search(chunks, vector_store, bm25, question)

                vector_chunks = [doc.page_content for doc in vector_results]
                combined_chunks = list(dict.fromkeys(vector_chunks + keyword_results))

                relevant_chunks = rerank(question, combined_chunks)
                context = "\n\n".join(relevant_chunks)
            else:
                # An empty context lets Groq answer as a general chatbot.
                context = ""

            answer = generate_answer(question, context)
        except GroqConfigurationError as error:
            st.error(str(error))
        except AuthenticationError:
            st.error("Groq rejected GROQ_API_KEY. Check that the key in .env is valid and active.")
        except NotFoundError:
            st.error("Groq does not recognize GROQ_MODEL. Set it in .env to an active model ID.")
        except RateLimitError:
            st.error("Groq rate limit or quota exceeded. Check your Groq account limits and try again.")
        except APIConnectionError:
            st.error("Could not connect to Groq. Check your internet connection and try again.")
        except APIStatusError as error:
            st.error(f"Groq rejected the request (HTTP {error.status_code}). Check the model and request settings.")
        else:
            st.write(answer)