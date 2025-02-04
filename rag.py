import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import streamlit as st


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
    llm = Ollama(temperature=0.6, model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
    )

    return qa_chain


def main():
    qa_chain = setup("resume.pdf")

    st.set_page_config(
        page_icon="💬",
        page_title="Deepseek RAG",
    )

    st.markdown(
        """
    # RAG powered by <img src="data:image/png;base64,{}" width="140" style="vertical-align: -3px;">
""".format(base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()),
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                    response_text += chunk["result"]
                    response_container.markdown(response_text + "▌")
                return response_text

            final_response = model_generator()
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )
            response_container.markdown(final_response)


if __name__ == "__main__":
    main()
