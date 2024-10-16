# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from utils.document_loader import load_pdf
from utils.text_preprocessor import preprocess_text
from utils.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store
)

# Ensure directories exist
os.makedirs('documents', exist_ok=True)
os.makedirs('vector_stores', exist_ok=True)

# Load environment variables for local development
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Operations Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e; /* Dark background color for the main content area */
    }
    .stSidebar {
        background-color: #2b2b2b; /* Dark background color for the sidebar */
    }
    .stButton button {
        background-color: #1E90FF; /* Button background color */
        color: #FFFFFF; /* Button text color */
    }
    .stTextInput input {
        background-color: #2b2b2b; /* Input field background color */
        color: #e0e0e0; /* Input field text color */
    }
    .css-1d391kg {  /* Adjust the header color */
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with instructions and document info
with st.sidebar:
    st.title("📄 Document Explorer")
    st.write("Hi there! I am **ZopA** 🤖 and I can help you understand and extract information from various documents. Simply choose a document and ask any questions you have.")
    st.write("---")
    st.subheader("Available Documents")
    # Get the list of available documents in the 'documents' directory
    available_files = [f for f in os.listdir('documents') if f.endswith(('.pdf', '.docx', '.txt'))]
    if available_files:
        selected_file = st.selectbox("Select a document to explore:", available_files)
        st.write("---")
        st.write("You can ask me anything about the selected document. Let's get started!")
    else:
        st.error("No documents found in the 'documents' directory. Please add PDF, DOCX, or TXT files to get started.")
        st.stop()

# Main content area
st.title("💬 Operations Assistant Chat")

if selected_file:
    # Function to load or create vector store
    def get_vector_store(file):
        index_folder_path = f'vector_stores/{os.path.splitext(file)[0]}_index'
        if os.path.exists(index_folder_path):
            # Load vector store from disk
            vector_store = load_vector_store(index_folder_path)
        else:
            # Process the document and create vector store
            document_path = os.path.join('documents', file)
            document_text = load_pdf(document_path)
            cleaned_document = preprocess_text(document_text)
            vector_store = create_vector_store(cleaned_document)
            # Save vector store to disk for future use
            save_vector_store(vector_store, index_folder_path)
        return vector_store

    with st.spinner(f"Loading and processing '{selected_file}'..."):
        try:
            # Use session state to cache the vector store
            if 'vector_store' not in st.session_state or st.session_state.get('current_file') != selected_file:
                st.session_state['vector_store'] = get_vector_store(selected_file)
                st.session_state['current_file'] = selected_file
                # Clear messages when a new document is loaded
                st.session_state['messages'] = []
            vector_store = st.session_state['vector_store']
            st.success(f"Document '{selected_file}' is ready!")
        except Exception as e:
            st.error(f"An error occurred while loading or processing the document: {e}")
            st.stop()

    # Initialize Language Model and QA Chain
    @st.cache_resource
    def initialize_llm():
        # Attempt to retrieve API key from Streamlit secrets
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            # Fallback to environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
                st.stop()
        return OpenAI(temperature=0, openai_api_key=api_key)

    llm = initialize_llm()

    def create_qa_chain(vector_store):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        return qa_chain

    if 'qa_chain' not in st.session_state or st.session_state.get('current_file') != selected_file:
        st.session_state['qa_chain'] = create_qa_chain(vector_store)
    qa_chain = st.session_state['qa_chain']

    # Initialize session state for conversation history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Chat interface using Streamlit's chat elements
    # Display conversation history
    for message in st.session_state['messages']:
        if message['user'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"{message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"{message['content']}")

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Append user's message to session state
        st.session_state['messages'].append({'user': 'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(f"{prompt}")

        with st.spinner("ZopA is typing..."):
            # Generate response from the QA chain
            response = qa_chain.run(prompt)
            # Append assistant's response to session state
            st.session_state['messages'].append({
                'user': 'assistant',
                'content': f"{response}\n\n😊 Thank you for your question! If you have any more questions or need further clarification, feel free to ask below."
            })

            with st.chat_message("assistant"):
                st.markdown(f"{response}\n\n😊 Thank you for your question! If you have any more questions or need further clarification, feel free to ask below.")

else:
    st.error("Please select a document from the sidebar to begin.")
