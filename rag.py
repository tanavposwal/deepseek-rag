import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


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
    llm = Ollama(temperature=0, model="llama3.2:3b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
    )

    return qa_chain


if __name__ == "__main__":
    qa_chain = setup("resume.pdf")

    while True:
        question = input("Ask a Question: ")
        if question.lower() == "exit":
            break

        answer = qa_chain.invoke(question)
        print("Answer:")
        print(answer["result"])
