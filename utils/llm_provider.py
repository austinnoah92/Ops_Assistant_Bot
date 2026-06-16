import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

# Provider configuration — display name and which secret key to look for
PROVIDER_CONFIG = {
    "OpenAI": {
        "key_name": "OPENAI_API_KEY",
        "display": "OpenAI (GPT-3.5)"
    },
    "Gemini": {
        "key_name": "GOOGLE_API_KEY",
        "display": "Google Gemini"
    },
    "Claude": {
        "key_name": "ANTHROPIC_API_KEY",
        "display": "Anthropic Claude"
    }
}

def get_api_key(key_name):
    """Retrieve API key from Streamlit secrets or environment variable."""
    try:
        key = st.secrets.get(key_name)
        if key:
            return key
    except Exception:
        pass
    return os.getenv(key_name)

def get_available_providers():
    """Return only providers whose API keys are actually present."""
    available = []
    for provider, config in PROVIDER_CONFIG.items():
        if get_api_key(config["key_name"]):
            available.append(provider)
    return available

def get_llm(provider):
    """Return the correct LLM object for the selected provider."""
    if provider == "OpenAI":
        return ChatOpenAI(
            temperature=0,
            openai_api_key=get_api_key("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=get_api_key("GOOGLE_API_KEY"),
            temperature=0
        )
    elif provider == "Claude":
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            anthropic_api_key=get_api_key("ANTHROPIC_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_embeddings(provider):
    """
    Return the correct embeddings object for the selected provider.
    """
    if provider == "OpenAI":
        return OpenAIEmbeddings(
            openai_api_key=get_api_key("OPENAI_API_KEY")
        )
    elif provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            output_dimensionality=768,
            batch_size=100,
            google_api_key=get_api_key("GOOGLE_API_KEY")
        )
    elif provider == "Claude":
        # Free local model — no API key required
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")