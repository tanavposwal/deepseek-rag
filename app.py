import base64
import os
import tempfile
import streamlit as st
from langchain.chains import RetrievalQA
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
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
    )

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
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header("Add Your PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # If there's a new file and we haven't set pdf_tool yet...
        if st.session_state.pdf_tool is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("Indexing PDF... Please wait..."):
                    qa_chain = setup(temp_file_path)

            st.success("PDF indexed! Ready to chat.")

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    st.button("Clear Chat", on_click=reset_chat)


def main():
    st.markdown(
        """
    # RAG powered by <img src="data:image/png;base64,{}" width="140" style="vertical-align: -3px;">
""".format(base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()),
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about PDF ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response_container = st.empty()

            def model_generator():
                stream = qa_chain.stream(prompt)
                response_text = ""
                for chunk in stream:
                    response_text += chunk["content"]
                    response_container.markdown(response_text + "▌")
                return response_text

            final_response = model_generator()
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )
            response_container.markdown(final_response)


if __name__ == "__main__":
    main()
