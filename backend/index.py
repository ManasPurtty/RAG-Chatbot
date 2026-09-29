"""
index.py - Batch indexing script for local documents into Qdrant Cloud.
Applies:
  - PII / Secret redaction
  - Fernet encryption
  - RBAC metadata tags (owner: admin, access_level: public)
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from security import redact_pii, encrypt_text

DATA_PATH = "../pdf"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "secure_rag_documents")

print("--- Secure RAG Batch Indexer ---")

# 1. Load documents
print("Loading text files...")
txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
txt_docs = txt_loader.load()

print("Loading PDF files...")
pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

all_docs = txt_docs + pdf_docs
print(f"Loaded {len(all_docs)} raw documents.")

if not all_docs:
    print("No documents found in", DATA_PATH)
    exit(0)

# 2. Split chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)
print(f"Created {len(chunks)} chunks.")

# 3. Apply Security: PII redaction, Fernet encryption, RBAC metadata
print("Applying PII redaction, encryption, and RBAC tags...")
for chunk in chunks:
    chunk.page_content = redact_pii(chunk.page_content)
    chunk.page_content = encrypt_text(chunk.page_content)
    chunk.metadata.update({
        "owner": "admin",
        "access_level": "public",
        "source_file": chunk.metadata.get("source", "batch_index"),
    })

# 4. Connect to Qdrant
if QDRANT_URL:
    print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("Connecting to local on-disk Qdrant at ../qdrant_db...")
    qdrant_client = QdrantClient(path="../qdrant_db")

# 5. Ensure collection exists
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    print(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
    )

# 6. Embed and store
print("Generating embeddings with BAAI/bge-small-en-v1.5 and uploading to Qdrant...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

vector_store.add_documents(chunks)
print(f"✅ Successfully indexed {len(chunks)} encrypted chunks to Qdrant collection '{QDRANT_COLLECTION}'!")
