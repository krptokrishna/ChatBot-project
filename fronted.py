import os
import streamlit as st

from ChatBot import get_response
from rag import build_vector_db

st.set_page_config(
    page_title="May I help you",
    page_icon="🤖",
    layout="wide"
)
# --------------------------
# Create Upload Folder
# --------------------------
os.makedirs("uploads", exist_ok=True)

# --------------------------
# Session State
# --------------------------

if "threads" not in st.session_state:
    st.session_state.threads = {
        "New Chat": []
    }

if "current_thread" not in st.session_state:
    st.session_state.current_thread = "New Chat"

# --------------------------
# Sidebar
# --------------------------

with st.sidebar:

    st.title("💬 Chats")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        chat_number = len(
            st.session_state.threads
        ) + 1

        new_thread = f"Chat {chat_number}"

        st.session_state.threads[
            new_thread
        ] = []

        st.session_state.current_thread = (
            new_thread
        )

        st.rerun()

    st.divider()

    # --------------------------
    # Chat List
    # --------------------------

    for thread_name in st.session_state.threads:

        if st.button(
            thread_name,
            key=f"thread_{thread_name}",
            use_container_width=True
        ):
            st.session_state.current_thread = (
                thread_name
            )

            st.rerun()

    st.divider()

    # --------------------------
    # PDF Upload
    # --------------------------

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )

        if st.button(
            "🧠 Build Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Building Knowledge Base..."
            ):

                success = build_vector_db()

            if success:

                st.success(
                    "Knowledge Base Created!"
                )

            else:

                st.error(
                    "No PDFs Found"
                )

    st.divider()

    # --------------------------
    # Uploaded Documents
    # --------------------------

    st.subheader("📚 Documents")

    pdf_files = [
        f
        for f in os.listdir("uploads")
        if f.endswith(".pdf")
    ]

    if pdf_files:

        for pdf in pdf_files:
            st.write(f"📄 {pdf}")

    else:

        st.caption(
            "No documents uploaded"
        )

# --------------------------
# Main Chat Area
# --------------------------

st.title("🤖 May I help you")

current_thread = (
    st.session_state.current_thread
)

messages = st.session_state.threads[
    current_thread
]

st.caption(
    f"Current Chat: {current_thread}"
)

# --------------------------
# Display Messages
# --------------------------

for msg in messages:

    with st.chat_message(
        msg["role"]
    ):
        st.write(
            msg["content"]
        )

# --------------------------
# Chat Input
# --------------------------

user_input = st.chat_input(
    "Type your message..."
)

if user_input:

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner(
        "Thinking..."
    ):

        response = get_response(
            user_input=user_input,
            thread_id=current_thread
        )

    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    st.session_state.threads[
        current_thread
    ] = messages

    st.rerun()