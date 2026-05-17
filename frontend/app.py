import os
import uuid
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AWS Docs Bot",
    page_icon="☁️",
    layout="centered",
)

st.title("☁️ AWS Documentation Assistant")
st.caption("Powered by Llama 3.3 70B via Amazon Bedrock • RAG on AWS Docs • LangChain Agents")

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Session")
    st.code(st.session_state.session_id[:8] + "...", language=None)
    if st.button("New Conversation", use_container_width=True):
        try:
            requests.delete(f"{BACKEND_URL}/chat/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What is Amazon S3 and when should I use it?",
        "Compare Lambda vs ECS vs EC2 for running containers",
        "How does VPC peering work?",
        "What are the best practices for IAM security?",
        "Explain AWS CDK vs CloudFormation vs Terraform",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_message = ex
            st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle pre-filled example click
if "pending_message" in st.session_state:
    prompt = st.session_state.pop("pending_message")
else:
    prompt = st.chat_input("Ask anything about AWS...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching AWS documentation..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.session_state.session_id = data["session_id"]
            except requests.exceptions.Timeout:
                answer = "Request timed out. The agent may be processing a complex query. Please try again."
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
