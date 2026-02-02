import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def download_and_ingest():
    pdf_filename = "nist_csf_clean.pdf"
    # Official NIST SP 800-61r3 IPD URL
    url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.ipd.pdf"
    
    # --- STEP 1: DOWNLOAD ---
    print(f"1. Downloading PDF from NIST...\n   URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Check for download errors
        with open(pdf_filename, 'wb') as f:
            f.write(response.content)
        print(f"   Success! Saved as '{pdf_filename}' ({len(response.content)} bytes).")
    except Exception as e:
        print(f"   CRITICAL ERROR: Could not download. {e}")
        return

    # --- STEP 2: LOAD ---
    print("2. Loading PDF...")
    try:
        loader = PyPDFLoader(pdf_filename)
        docs = loader.load()
        print(f"   Loaded {len(docs)} pages.")
    except Exception as e:
        print(f"   Error reading PDF: {e}")
        return

    # --- STEP 3: SPLIT ---
    print("3. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    if not splits:
        print("   Error: No text found in PDF.")
        return
        
    print(f"   Created {len(splits)} text chunks.")

    # --- STEP 4: EMBED ---
    print("4. Creating Database (this takes ~1 min)...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear old DB if it exists (optional but safer)
    if os.path.exists("./chroma_db"):
        import shutil
        try:
            shutil.rmtree("./chroma_db")
            print("   (Removed old database to start fresh)")
        except:
            pass

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    
    print("5. DONE! Database is ready in './chroma_db'")

if __name__ == "__main__":
    download_and_ingest()