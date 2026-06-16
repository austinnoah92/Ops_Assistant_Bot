import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load your API keys from .env
load_dotenv()

def generate_ai_documents():
    os.makedirs('documents', exist_ok=True)
    
    print("Initializing LLM...")
    # Using Gemini model to generate document
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # Prompt template telling the LLM what and how to write
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical writer for an enterprise SaaS company."),
        ("human", "Write a comprehensive {doc_type} document. It should be highly detailed, professional, and contain sections on {sections}.")
    ])
    
    chain = prompt | llm
    
    # --- 1. Draft the Support Document ---
    print("Drafting Customer Support SLA Document...")
    support_result = chain.invoke({
        "doc_type": "Customer Support SLA",
        "sections": "Response Times, Escalation Protocols, Tier 1/2/3 definitions, and compliance rules"
    })
    
    with open('documents/AI_Support_Manual.txt', 'w', encoding='utf-8') as f:
        f.write(support_result.content)
        
    # --- 2. Draft the Product Document ---
    print("Drafting Product Architecture Document...")
    product_result = chain.invoke({
        "doc_type": "Product Architecture Manual",
        "sections": "Data Ingestion API, Statistical Compute Cluster capabilities, Deployment protocols, and error troubleshooting"
    })
    
    with open('documents/AI_Product_Manual.txt', 'w', encoding='utf-8') as f:
        f.write(product_result.content)
        
    print("\nSuccess! AI-generated documents saved to the 'documents' folder.")

if __name__ == "__main__":
    generate_ai_documents()