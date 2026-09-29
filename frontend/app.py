"""
app.py - Secure RAG Chatbot Frontend

Features added:
  ✅ Login / Logout UI with JWT token management
  ✅ Token sent in Authorization header on every request
  ✅ Role displayed in sidebar (admin / user)
  ✅ Access level selector for uploads (admin only sees "public" option)
  ✅ Security info displayed per query response
"""

import streamlit as st
import requests
import os

BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
LOGIN_URL   = f"{BASE_URL}/login"
UPLOAD_URL  = f"{BASE_URL}/upload"
QUERY_URL   = f"{BASE_URL}/query"
ME_URL      = f"{BASE_URL}/me"

st.set_page_config(page_title="Secure RAG Chatbot", page_icon="🔒", layout="wide")


# ── Session defaults ────────────────────────────────────────────────────────────
for key, default in {
    "token": None,
    "username": None,
    "role": None,
    "messages": [],
    "uploader_key": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def auth_header() -> dict:
    """Return the Authorization header dict for API calls."""
    return {"Authorization": f"Bearer {st.session_state.token}"}


def logout():
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.messages = []
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.token:
    st.title("🔒 Secure RAG Chatbot")
    st.subheader("Please log in to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            try:
                resp = requests.post(
                    LOGIN_URL,
                    json={"username": username, "password": password},
                    timeout=15,
                )
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        st.error(f"Backend returned non-JSON (status 200). Body: {resp.text[:300]}")
                        st.stop()

                    st.session_state.token = data["access_token"]

                    me_resp = requests.get(ME_URL, headers=auth_header(), timeout=10)
                    if me_resp.status_code == 200:
                        me = me_resp.json()
                        st.session_state.username = me["username"]
                        st.session_state.role = me["role"]
                    else:
                        st.error(f"/me failed (HTTP {me_resp.status_code}): {me_resp.text[:200]}")
                        st.stop()

                    st.success("Login successful!")
                    st.rerun()
                else:
                    try:
                        detail = resp.json().get("detail", resp.text[:200])
                    except Exception:
                        detail = resp.text[:200]
                    st.error(f"Login failed (HTTP {resp.status_code}): {detail}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach backend at `http://127.0.0.1:8000`. Is uvicorn running?")
            except requests.exceptions.Timeout:
                st.error("❌ Backend timed out — it may still be loading the embedding model (wait ~60s and retry).")
            except Exception as e:
                st.error(f"Unexpected error — {type(e).__name__}: {e}")

    st.info("💡 Default credentials — username: `admin` | password: `admin123`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (authenticated)
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔒 Secure RAG")
    st.markdown(f"**User:** `{st.session_state.username}`")
    role_badge = "🛡️ Admin" if st.session_state.role == "admin" else "👤 User"
    st.markdown(f"**Role:** {role_badge}")
    st.divider()

    # ── Upload section ──────────────────────────────────────────────────────────
    st.subheader("📄 Upload PDF")

    access_options = ["private"]
    if st.session_state.role == "admin":
        access_options = ["private", "public"]

    access_level = st.selectbox(
        "Access Level",
        access_options,
        help="Private: only you (and admins) can query this document.\nPublic: all users can query it (admin only).",
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file:
        if st.button("Upload & Index"):
            with st.spinner("Uploading, encrypting, and indexing…"):
                try:
                    resp = requests.post(
                        f"{UPLOAD_URL}?access_level={access_level}",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        headers=auth_header(),
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ Indexed {data['chunks_added']} chunks  "
                            f"| Owner: `{data['owner']}`  "
                            f"| Level: `{data['access_level']}`"
                        )
                        st.session_state.uploader_key += 1
                        st.rerun()
                    elif resp.status_code == 401:
                        st.error("Session expired. Please log in again.")
                        logout()
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()
    if st.button("🚪 Logout"):
        logout()

# ── Chat interface ──────────────────────────────────────────────────────────────
st.title("🔒 Secure RAG Chatbot")
st.caption("Your queries are PII-redacted · Retrieved context is encrypted at rest · Injection-protected")

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New question
user_prompt = st.chat_input("Ask a question about your documents…")

if user_prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    QUERY_URL,
                    json={"text": user_prompt},
                    headers=auth_header(),
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer  = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])
                    security_info = data.get("security", {})

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Sources"):
                            st.json(sources)

                    if security_info:
                        with st.expander("🔐 Security Info"):
                            for k, v in security_info.items():
                                icon = "✅" if v else "➖"
                                st.markdown(f"{icon} **{k.replace('_', ' ').title()}**: `{v}`")

                elif resp.status_code == 400:
                    detail = resp.json().get("detail", "Bad request.")
                    answer = f"⚠️ {detail}"
                    st.warning(answer)

                elif resp.status_code == 401:
                    answer = "🔒 Session expired. Please log in again."
                    st.error(answer)
                    logout()

                else:
                    answer = f"Backend error ({resp.status_code}): {resp.text[:200]}"
                    st.error(answer)

            except Exception as e:
                answer = f"Connection error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})