import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"
UPLOAD_URL = "http://127.0.0.1:8000/upload"
st.title("RAG Chatbot")
st.write("Ask Question about your document")

st.divider()

st.subheader("Upload PDF")
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state.uploader_key}"
)


if uploaded_file:

    if st.button("Upload"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        response = requests.post(
            UPLOAD_URL,
            files=files
        )

        if response.status_code == 200:
            st.success("PDF uploaded successfully!")
        else:
            st.error("Upload failed.")

if "messages" not in st.session_state:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:

        response = requests.post(
            API_URL,
            json={"text": prompt},
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("answer", "No answer found.")
        sources = data.get("sources", [])

    except Exception as e:

        answer = f"Backend Error: {e}"
        sources = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:

            with st.expander("Sources"):
                st.json(sources)