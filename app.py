import os
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.env_sync import sync_env_to_secrets
from utils.document_loader import load_pdf
from utils.text_preprocessor import preprocess_text
from utils.vector_store import create_vector_store, load_vector_store, save_vector_store
from utils.llm_provider import get_available_providers, get_llm, get_embeddings

# Load environment variables for local development
sync_env_to_secrets()
load_dotenv()

# Ensure directories exist
os.makedirs('documents', exist_ok=True)
os.makedirs('vector_stores', exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Operations Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; }
    .stSidebar { background-color: #2b2b2b; }
    .stButton button { background-color: #1E90FF; color: #FFFFFF; }
    .stTextInput input { background-color: #2b2b2b; color: #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def get_cached_vector_store(file, provider):
    """
    Load existing vector store for this file+provider combination,
    or create and save a new one. Cached to prevent reloading on every interaction.
    """
    index_folder_path = f'vector_stores/{os.path.splitext(file)[0]}_{provider}_index'
    embeddings = get_embeddings(provider)
    
    if os.path.exists(index_folder_path):
        vector_store = load_vector_store(index_folder_path, embeddings)
    else:
        document_path = os.path.join('documents', file)
        document_text = load_pdf(document_path)
        cleaned_document = preprocess_text(document_text)
        vector_store = create_vector_store(cleaned_document, embeddings)
        save_vector_store(vector_store, index_folder_path)
    return vector_store

@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(_vector_store, provider):
    """Cached function to build the retrieval chain with a conversational prompt."""
    llm = get_llm(provider)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Operations Assistant. Answer the question using only the context provided. "
         "If the answer is not in the context, politely acknoweldge the user questions and give reasons why you are unable to answer.\n\n"
         "CRITICAL INSTRUCTION: At the very end of your response, always naturally suggest a specific "
         "direction we could explore next, or ask a relevant follow-up question based on the document's content. "
         "Make it feel conversational and engaging.\n\n"
         "Context: {context}"),
        ("human", "{input}")
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        _vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_chain
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Explorer")
    st.write(
        "Hi there! I am **ZopA** 🤖 and I can help you understand "
        "and extract information from various documents. "
        "Choose a document and a provider, then ask your questions."
    )
    st.write("---")

    # Provider selection — only shows providers with keys available
    available_providers = get_available_providers()
    if not available_providers:
        st.error(
            "No LLM API keys found. Please add at least one of: "
            "OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY "
            "to your .env file or Streamlit secrets."
        )
        st.stop()

    st.subheader("Select AI Provider")
    selected_provider = st.selectbox(
        "Provider:",
        available_providers,
        help="Only providers with valid API keys are shown."
    )
    st.write("---")

    # Document selection
    st.subheader("Available Documents")
    available_files = [
        f for f in os.listdir('documents')
        if f.endswith(('.pdf', '.docx', '.txt'))
    ]
    if available_files:
        selected_file = st.selectbox(
            "Select a document to explore:", available_files
        )
        st.write("---")
        st.write(
            "You can ask me anything about the selected document. "
            "Let's get started!"
        )
    else:
        st.error(
            "No documents found in the 'documents' directory. "
            "Please add PDF, DOCX, or TXT files to get started."
        )
        st.stop()

# ── Main content ──────────────────────────────────────────────────────────────
st.title("💬 Operations Assistant Chat")

if selected_file:
    with st.spinner(f"Loading '{selected_file}' with {selected_provider}..."):
        try:
            vector_store = get_cached_vector_store(selected_file, selected_provider)
            qa_chain = get_cached_qa_chain(vector_store, selected_provider)
        except Exception as e:
            st.error(f"Error loading document: {e}")
            st.stop()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display conversation history
    for message in st.session_state['messages']:
        role = "user" if message['user'] == 'user' else "assistant"
        with st.chat_message(role):
            st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        st.session_state['messages'].append(
            {'user': 'user', 'content': prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("ZopA is typing..."):
            result = qa_chain.invoke({"input": prompt})
            response = result["answer"]
            
            st.session_state['messages'].append(
                {'user': 'assistant', 'content': response}
            )
            with st.chat_message("assistant"):
                st.markdown(response)
else:
    st.error("Please select a document from the sidebar to begin.")
