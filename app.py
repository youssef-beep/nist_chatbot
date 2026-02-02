import streamlit as st
import json
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NIST AI Assistant", page_icon="🛡️")
st.title("🛡️ NIST SP 800-61r3 Assistant")
st.markdown("Internal Incident Response Knowledge Base")

# --- 2. LOAD RESOURCES (Cached so it's fast) ---
@st.cache_resource
def load_system():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    llm = OllamaLLM(model="llama3.2")
    return db, llm

db, llm = load_system()

# --- 3. LOGIC FUNCTIONS ---
def get_router_decision(question):
    prompt_text = """
    You are a routing system. Output ONLY JSON: {{"mode": "rag"}} or {{"mode": "clarify"}}.
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    response = llm.invoke(prompt.format(question=question))
    try:
        start = response.find("{")
        end = response.find("}") + 1
        return json.loads(response[start:end]).get("mode", "clarify")
    except:
        return "clarify"

def run_rag_pipeline(question):
    docs = db.similarity_search(question, k=3)
    context_text = "\n\n".join([f"(Page {d.metadata.get('page')}) {d.page_content}" for d in docs])
    
    prompt_text = """
    You are a NIST Assistant. Answer using ONLY this context. Cite [Page X].
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return llm.invoke(prompt.format(context=context_text, question=question))

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about NIST Incident Response..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing NIST Guidelines..."):
            # Step 1: Route
            mode = get_router_decision(prompt)
            
            if mode == "rag":
                response = run_rag_pipeline(prompt)
                st.markdown(response)
                st.info("Source: NIST SP 800-61r3 (Local Vector DB)")
            else:
                response = "I am a specialized NIST assistant. Please ask a question related to Cybersecurity Incident Response."
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- 5. SIDEBAR INFO ---
st.sidebar.title("System Status")
st.sidebar.success("Ollama: Llama 3.2 Active")
st.sidebar.success("DB: Chroma (NIST SP 800-61r3)")
st.sidebar.write("---")
st.sidebar.markdown("""
**How it works:**
1. **Router:** Decides if the query is on-topic.
2. **RAG:** Retrieves chunks from the local vector database.
3. **LLM:** Generates response with citations.
""")