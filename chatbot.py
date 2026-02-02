import json
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
llm = OllamaLLM(model="llama3.2")

# --- PROMPT 1: THE ROUTER ---
ROUTER_PROMPT = """
You are a routing system. Determine if the user's question is about cybersecurity incident response or NIST guidelines.
You must output ONLY a JSON object and nothing else.

JSON Format: {{"mode": "rag"}} OR {{"mode": "clarify"}}

- If it's a specific question about cybersecurity: {{"mode": "rag"}}
- If it's a greeting, nonsense, or totally unrelated: {{"mode": "clarify"}}

User Question: {question}
"""

# --- THE "FINE-TUNED" SYSTEM PROMPT ---
# We add a "Style Example" section here to mimic a fine-tuned model.
ANSWER_PROMPT = """
You are a NIST Certified Assistant. You must follow the "NIST Style" perfectly.

NIST STYLE RULES:
1. Start every response with "NIST GUIDANCE:"
2. Use bullet points for lists.
3. Always cite the page in brackets at the end of the sentence like [Page X].
4. Be extremely formal.

### EXAMPLE OF PERFECT STYLE ###
User: What is a cybersecurity incident?
NIST GUIDANCE:
An occurrence that actually or imminently jeopardizes the integrity, confidentiality, or availability of information [Page 39].

### CURRENT TASK ###
Context: {context}
Question: {question}

Answer:
"""

def get_router_decision(question):
    """This function acts as the gatekeeper."""
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    formatted_prompt = prompt.format(question=question)
    response = llm.invoke(formatted_prompt)
    
    # We try to clean up the response in case the AI adds extra words
    try:
        # Look for the JSON part
        start = response.find("{")
        end = response.find("}") + 1
        data = json.loads(response[start:end])
        return data.get("mode", "clarify")
    except:
        return "clarify"

def run_rag_pipeline(question):
    """The standard search-and-answer process."""
    docs = db.similarity_search(question, k=3)
    context_text = "\n\n".join([f"(Page {d.metadata.get('page')}) {d.page_content}" for d in docs])
    
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
    final_prompt = prompt.format(context=context_text, question=question)
    return llm.invoke(final_prompt)

def chatbot():
    print("AI Assistant Active! (Type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit': break
        
        print("...AI is thinking...")
        
        # --- THE PROMPT ENGINEERING STEP ---
        mode = get_router_decision(user_input)
        
        if mode == "rag":
            print("[System: Routing to Database Search]")
            answer = run_rag_pipeline(user_input)
            print(f"\nAI: {answer}")
        else:
            print("[System: Requesting Clarification]")
            print("\nAI: I'm a specialized NIST Assistant. Could you please ask a question related to Cybersecurity Incident Response?")

if __name__ == "__main__":
    chatbot()