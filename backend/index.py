#load split embeddings store
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
from langchain_community.document_loaders import DirectoryLoader ,PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
DATA_PATH="../pdf"
CHROMA_PATH = "../chroma_db"
print("Loading text files")
txt_loader=DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)

txt_docs=txt_loader.load()
pdf_loader=DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs=pdf_loader.load()
docs=txt_docs+pdf_docs
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs=text_splitter.split_documents(docs)
print("Creating embeddings")
embeddings = HuggingFaceEmbeddings(
model_name="BAAI/bge-small-en-v1.5")
print("Creating vector store")
vector_store=Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
vector_store.persist()
print("CHROMA INDEX CREATED AND SAVED")












