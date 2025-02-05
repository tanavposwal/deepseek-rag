import os
import re
import base64
import tempfile
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(
    page_icon="💬",
    page_title="Deepseek RAG",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None


def setup(file_path):
    """
    Setup the conversational retrieval chain by loading a PDF, splitting its content,
    embedding it, and creating a vector store-based retriever.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever()
    llm = Ollama(temperature=0.6, model="deepseek-r1")

    # Create a conversational retrieval chain to support context.
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain


def reset_chat():
    st.session_state.messages = []


def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    ></iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)


def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )


def process_streaming_response(stream):
    """
    Process the streaming response from the assistant.
    It collects tokens in a single pass, first handling the thinking phase (delimited by <think> tags)
    and then the response phase.
    """
    thinking_content = ""
    response_content = ""
    phase = "thinking"
    think_placeholder = st.empty()
    response_placeholder = st.empty()

    with st.spinner("Thinking..."):
        for chunk in stream:
            # Assumes the streaming chunk is a dict with a nested "message" dict.
            content = chunk.get("message", {}).get("content", "")
            if phase == "thinking":
                thinking_content += content
                if "</think>" in content:
                    phase = "response"
                    formatted_thinking = format_reasoning_response(thinking_content)
                    with st.expander("Thinking complete!"):
                        st.markdown(formatted_thinking)
                else:
                    think_placeholder.markdown(
                        format_reasoning_response(thinking_content)
                    )
            else:
                response_content += content
                response_placeholder.markdown(response_content)

    return thinking_content, response_content


def display_assistant_message(content):
    """
    Display the assistant message.
    If the content includes <think> tags, show the reasoning part in an expander.
    """
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        think_content = think_match.group(0)
        response_content = content.replace(think_content, "")
        formatted_think = format_reasoning_response(think_content)
        with st.expander("Thinking complete!"):
            st.markdown(formatted_think)
        st.markdown(response_content)
    else:
        st.markdown(content)


def convert_chat_history(messages):
    """
    Convert the conversation history (a list of messages with roles "user" and "assistant")
    into a list of (question, answer) tuples as expected by the conversational chain.
    Only complete pairs are returned.
    """
    history = []
    user_msg = None
    for message in messages:
        if message["role"] == "user":
            user_msg = message["content"]
        elif message["role"] == "assistant" and user_msg is not None:
            history.append((user_msg, message["content"]))
            user_msg = None
    return history


def main():
    st.markdown(
        f"""
    # RAG powered by <img src="data:image/png;base64,{base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()}" width="140" style="vertical-align: -3px;">
    """,
        unsafe_allow_html=True,
    )

    # Display previous chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar: PDF uploader, indexing, and preview.
    with st.sidebar:
        st.header("Add Your PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            # If a new file is uploaded (or when first uploaded), initialize/re-index the document.
            if st.session_state.get("pdf_tool") is None:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    with st.spinner("Indexing PDF... Please wait..."):
                        qa_chain = setup(temp_file_path)
                        st.session_state.pdf_tool = qa_chain
                st.success("PDF indexed! Ready to chat.")
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
        st.button("Clear Chat", on_click=reset_chat)

    # Chat input.
    if prompt := st.chat_input("Ask a question about PDF ..."):
        # Append user prompt to history.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            qa_chain = st.session_state.get("pdf_tool")
            if qa_chain is None:
                st.error("Please upload a PDF document first.")
                return

            # For conversational context, pass complete conversation pairs (excluding the current pending question if needed).
            chat_history = convert_chat_history(st.session_state.messages[:-1])
            try:
                # Call the chain with the current question and conversation history.
                result = qa_chain({"question": prompt, "chat_history": chat_history})
                # If the chain returns a dictionary with "answer", assume non-streaming mode.
                if isinstance(result, dict) and "answer" in result:
                    full_response = result["answer"]
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    display_assistant_message(full_response)
                else:
                    # Otherwise, assume streaming and process the stream.
                    thinking_content, response_content = process_streaming_response(
                        result
                    )
                    full_response = f"{thinking_content}{response_content}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    display_assistant_message(full_response)
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
