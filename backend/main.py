import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi import UploadFile, File
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

CHROMA_PATH = "../chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
      max_output_tokens=1024, 
      google_api_key=os.getenv("GEMINI_API_KEY"))
retriever = db.as_retriever(search_kwargs={"k": 3})
SYSTEM_PROMPT = """
You are an AI assistant.

Answer ONLY from the provided context.

If the answer is not found in the context,
reply:

"I don't know based on the provided documents."

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])



app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    text: str

@app.post("/query")
async def query_rag(query:Query):

    docs = retriever.invoke(query.text)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = prompt.invoke(
        {
            "context": context,
            "question": query.text,
        }
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in docs],
    }
UPLOAD_FOLDER = "../pdf"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Save PDF
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load uploaded PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Add chunks into existing ChromaDB
    db.add_documents(chunks)

    # Save updated database
    db.persist()

    return {
        "message": "PDF uploaded and indexed successfully.",
        "chunks_added": len(chunks)
    }