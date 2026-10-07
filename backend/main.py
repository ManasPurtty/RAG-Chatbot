"""
main.py - Secure RAG Chatbot Backend

Security features implemented:
  ✅ JWT Authentication         — every endpoint requires a valid token
  ✅ RBAC                       — admins see all docs; users see their own + public
  ✅ Metadata-filtered retrieval — ChromaDB where-filter enforces access rules
  ✅ PII / Secret Redaction      — user input and LLM output are both scrubbed
  ✅ Context Minimisation        — caps context chars sent to the LLM (max 3000)
  ✅ Encrypted Storage           — page_content encrypted with Fernet before storage
  ✅ Secure Logging              — sensitive fields masked in all log output
  ✅ Output Leakage Checks       — LLM response scanned for PII before returning
  ✅ Prompt Injection Protection — retrieved docs wrapped as untrusted data
"""

import os
import tempfile
from datetime import timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Local modules ──────────────────────────────────────────────────────────────
from models import Query, Token, LoginRequest, RegisterRequest, UserRole, UserInDB
from auth import (
    authenticate_user,
    create_access_token,
    get_password_hash,
    get_current_user,
    require_admin,
    load_users,
    save_users,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from security import (
    secure_log,
    sanitize_user_input,
    redact_pii,
    redact_output,
    check_output_leakage,
    minimize_context,
    encrypt_text,
    decrypt_text,
    build_safe_system_prompt,
)

# ── Vector DB & LLM Init ───────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "secure_rag_documents")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

if QDRANT_URL:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        port=443,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )
    secure_log("QDRANT_INIT", host=QDRANT_URL)
else:
    qdrant_client = QdrantClient(path="../qdrant_db")
    secure_log("QDRANT_INIT", mode="local_path")

# Ensure collection exists (dimension=384 for bge-small-en-v1.5)
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=1024,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Secure RAG Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/login", response_model=Token)
async def login(form_data: LoginRequest):
    """
    Authenticate user and return a JWT access token.
    Uses constant-time comparison to prevent username enumeration.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        secure_log("LOGIN_FAILED", username=form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    secure_log("LOGIN_SUCCESS", username=user.username, role=user.role)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/register", status_code=201)
async def register(
    payload: RegisterRequest,
    admin: UserInDB = Depends(require_admin),
):
    """
    Register a new user. Admin-only endpoint.
    """
    users = load_users()
    if payload.username in users:
        raise HTTPException(status_code=400, detail="Username already exists.")
    users[payload.username] = {
        "username": payload.username,
        "hashed_password": get_password_hash(payload.password),
        "role": payload.role.value,
    }
    save_users(users)
    secure_log("USER_REGISTERED", username=admin.username, new_user=payload.username, role=payload.role)
    return {"message": f"User '{payload.username}' created with role '{payload.role}'."}


@app.get("/me")
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    """Return current authenticated user's profile."""
    return {"username": current_user.username, "role": current_user.role}


@app.get("/users")
async def list_users(admin: UserInDB = Depends(require_admin)):
    """List all users. Admin-only."""
    users = load_users()
    return [
        {"username": u["username"], "role": u["role"]}
        for u in users.values()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/query")
async def query_rag(
    query: Query,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    RAG query endpoint with full security pipeline:
    1. Prompt injection detection on user input
    2. PII redaction on user input
    3. RBAC-filtered vector retrieval (metadata where-filter)
    4. Context minimisation
    5. Encrypted chunk decryption
    6. Safe system prompt (untrusted context wrapper)
    7. LLM invocation
    8. Output leakage check + PII redaction on response
    """

    # ── Step 1: Detect prompt injection ───────────────────────────────────────
    user_text, injection_detected = sanitize_user_input(query.text)
    if injection_detected:
        secure_log("INJECTION_ATTEMPT", username=current_user.username, query_snippet=query.text[:80])
        raise HTTPException(
            status_code=400,
            detail="Query rejected: potential prompt injection detected.",
        )

    # ── Step 2: Redact PII from user question before logging / sending ─────────
    sanitized_question = redact_pii(user_text)
    secure_log("QUERY_RECEIVED", username=current_user.username)

    # ── Step 3: RBAC-filtered retrieval ───────────────────────────────────────
    # Admins see everything; regular users see only public + their own docs.
    if current_user.role == UserRole.admin:
        qdrant_filter = None  # no restriction
    else:
        qdrant_filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="metadata.access_level",
                    match=qmodels.MatchValue(value="public"),
                ),
                qmodels.FieldCondition(
                    key="metadata.owner",
                    match=qmodels.MatchValue(value=current_user.username),
                ),
            ]
        )

    raw_docs = vector_store.similarity_search(
        sanitized_question,
        k=5,
        filter=qdrant_filter,
    )

    # ── Step 4: Context minimisation ──────────────────────────────────────────
    docs = minimize_context(raw_docs, max_chars=3000)
    secure_log("RETRIEVAL_DONE", username=current_user.username, chunks_retrieved=len(docs))

    # ── Step 5: Decrypt stored content ────────────────────────────────────────
    decrypted_parts = []
    for doc in docs:
        plain = decrypt_text(doc.page_content)
        decrypted_parts.append(plain)

    context = "\n\n".join(decrypted_parts)

    # ── Step 6: Build hardened system prompt ──────────────────────────────────
    system_content = build_safe_system_prompt(context, current_user.username)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=sanitized_question),
    ]

    # ── Step 7: LLM invocation ─────────────────────────────────────────────────
    response = llm.invoke(messages)
    raw_answer = response.content

    # ── Step 8: Output leakage check + redaction ───────────────────────────────
    leakage = check_output_leakage(raw_answer)
    if leakage:
        secure_log("OUTPUT_LEAKAGE_DETECTED", username=current_user.username, pii_types=list(leakage.keys()))
        raw_answer = redact_output(raw_answer)

    secure_log("QUERY_ANSWERED", username=current_user.username)

    # Sanitise source metadata before returning (strip encrypted blobs)
    safe_sources = []
    for doc in docs:
        meta = {k: v for k, v in doc.metadata.items() if k != "page_content"}
        safe_sources.append(meta)

    return {
        "answer": raw_answer,
        "sources": safe_sources,
        "security": {
            "injection_checked": True,
            "pii_redacted": True,
            "output_leakage_check": bool(leakage),
            "context_minimised": len(docs) < len(raw_docs),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    access_level: str = "private",          # "public" or "private"
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Upload and index a PDF with:
    - Owner & access_level metadata tagging (RBAC)
    - Fernet encryption of each chunk before storage
    - PII redaction on chunk text before encrypting
    - Only admins can upload public documents
    - Processing fully in-memory (no permanent disk write)
    """

    # ── Validate access level ─────────────────────────────────────────────────
    if access_level not in ("public", "private"):
        raise HTTPException(status_code=400, detail="access_level must be 'public' or 'private'.")
    if access_level == "public" and current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Only admins can upload public documents.")

    secure_log("UPLOAD_START", username=current_user.username, filename=file.filename, access_level=access_level)

    # ── Read PDF in memory, write to temp file for PyPDFLoader ────────────────
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    # ── Chunk documents ────────────────────────────────────────────────────────
    chunks = splitter.split_documents(raw_docs)

    # ── Encrypt + tag each chunk ───────────────────────────────────────────────
    for chunk in chunks:
        # 1. Redact PII from chunk text
        chunk.page_content = redact_pii(chunk.page_content)
        # 2. Encrypt chunk text before storage
        chunk.page_content = encrypt_text(chunk.page_content)
        # 3. Attach RBAC metadata
        chunk.metadata.update({
            "owner": current_user.username,
            "access_level": access_level,
            "source_file": file.filename,
        })

    # ── Add to Qdrant Vector Store ─────────────────────────────────────────────
    vector_store.add_documents(chunks)

    secure_log("UPLOAD_DONE", username=current_user.username, filename=file.filename, chunks=len(chunks))

    return {
        "message": "PDF uploaded, encrypted, and indexed successfully.",
        "chunks_added": len(chunks),
        "owner": current_user.username,
        "access_level": access_level,
    }