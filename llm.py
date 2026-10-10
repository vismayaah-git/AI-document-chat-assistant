import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

load_dotenv()


class GroqConfigurationError(ValueError):
    """Raised when the Groq credentials or model configuration is missing."""


@st.cache_resource
def get_client(api_key):
    return Groq(api_key=api_key)


def generate_answer(question, context):
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if not api_key:
        raise GroqConfigurationError("GROQ_API_KEY is missing. Add it to the project .env file.")
    api_key = api_key.strip()
    if "your_" in api_key.lower() or "paste" in api_key.lower():
        raise GroqConfigurationError(
            "GROQ_API_KEY is still a placeholder. Create a Groq API key and set it "
            "in the local .env file; do not share or commit it."
        )
    if not model:
        raise GroqConfigurationError("GROQ_MODEL is missing. Add it to the project .env file.")
    model = model.strip()
    if model.lower().startswith("your_"):
        raise GroqConfigurationError("GROQ_MODEL is still a placeholder. Set it to an active Groq model ID.")

    response = get_client(api_key).chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion:\n{question}" }], temperature=0.2)
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("The language model returned an empty response.")
    return answer