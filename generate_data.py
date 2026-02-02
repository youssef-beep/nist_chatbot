import json
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Setup
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
llm = OllamaLLM(model="llama3.2")

def create_synthetic_data():
    print("Generating training examples... please wait.")
    
    # Get 10 random chunks from your database
    # (In a real project, you would do 100+, but we'll start with 10 for speed)
    all_docs = db.get()
    training_data = []

    for i in range(10):
        content = all_docs['documents'][i]
        page = all_docs['metadatas'][i].get('page', 'Unknown')
        
        # This prompt tells the AI to create a training example
        gen_prompt = f"""
        Based on the NIST text below, create:
        1. A user question.
        2. A perfect, professional answer citing Page {page}.
        
        Text: {content}
        
        Output in this EXACT JSON format:
        {{"instruction": "user question here", "context": "NIST text here", "response": "perfect answer here"}}
        """
        
        raw_response = llm.invoke(gen_prompt)
        
        try:
            # Clean and save the JSON
            start = raw_response.find("{")
            end = raw_response.find("}") + 1
            example = json.loads(raw_response[start:end])
            training_data.append(example)
            print(f"Generated example {i+1}/10")
        except:
            continue

    # Save to a file
    with open("fine_tuning_data.jsonl", "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")
            
    print("\nDONE! You now have 'fine_tuning_data.jsonl' with your training examples.")

if __name__ == "__main__":
    create_synthetic_data()