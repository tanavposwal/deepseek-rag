import os
import base64
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(
    page_icon="💬",
    page_title="Deepseek RAG",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None


def setup(file_path):
    """Set up the RAG pipeline with the uploaded PDF."""
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    llm = ChatOllama(
        temperature=0.6, model="smollm2", streaming=True
    )  # Enable streaming

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff"
    )

    return qa_chain


def reset_chat():
    """Reset the chat history."""
    st.session_state.messages = []


def display_pdf(file_bytes: bytes, file_name: str):
    """Display the uploaded PDF in an iframe."""
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


def display_sidebar():
    """Display the sidebar for uploading PDFs and clearing the chat."""
    with st.sidebar:
        st.header("Add Your PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
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


def main():
    """Main function to run the Streamlit app."""
    st.markdown(
        f"""
    # RAG powered by <img src="data:image/png;base64,{base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()}" width="140" style="vertical-align: -3px;">
    """,
        unsafe_allow_html=True,
    )

    display_sidebar()

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            qa_chain = st.session_state.get("pdf_tool")
            if qa_chain is None:
                st.error("Please upload a PDF document first.")
                return

            try:
                # Get the response from the QA chain
                response = qa_chain.invoke({"query": prompt})
                if isinstance(response, dict) and "result" in response:
                    # Stream the result text
                    result_text = response["result"]
                    st.write(result_text)
                else:
                    st.error("Unexpected response format.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
