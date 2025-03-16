import os
import base64
import tempfile
import streamlit as st

# import ollama
from crewai import LLM
from crewai_tools import PDFSearchTool
from crews import init_crew


@st.cache_resource
def get_llm():
    """Cache the LLM initialization"""
    return LLM(model="ollama/gemma3", base_url="http://localhost:11434")


# models = [model["model"] for model in ollama.list()["models"]]

# Page configuration
st.set_page_config(
    page_icon="🚀",
    page_title="Crew AI RAG",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "crew" not in st.session_state:
    st.session_state.crew = None
# if "model" not in st.session_state:
#     st.session_state["model"] = None


def setup(file_path):
    rag_tool = PDFSearchTool(
        pdf=file_path,
        config=dict(
            llm=dict(
                provider="ollama",
                config=dict(model="gemma3"),
            ),
            embedder=dict(
                provider="ollama",
                config=dict(model="nomic-embed-text"),
            ),
        ),
    )
    st.session_state.crew = init_crew(rag_tool)
    return rag_tool


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
        height="400px" 
        type="application/pdf"
    ></iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar for uploading PDFs and clearing the chat."""
    with st.sidebar:
        # st.session_state["model"] = st.selectbox("Choose your model", models)
        st.header("Add Your PDF Document")

        # Add container for upload status
        upload_container = st.container()

        with upload_container:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                try:
                    if st.session_state.get("crew") is None:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            with st.spinner(
                                "📚 Indexing PDF... This may take a moment"
                            ):
                                tool = setup(temp_file_path)
                                st.session_state.pdf_tool = tool

                        st.success("✅ PDF indexed! Ready to chat.")
                    display_pdf(uploaded_file.getvalue(), uploaded_file.name)
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.crew = None
                    st.session_state.pdf_tool = None

        # Add divider for better visual separation
        st.divider()

        # Add clear chat button with confirmation
        if st.button("🗑️ Clear Chat History", type="secondary"):
            if len(st.session_state.messages) > 0:
                if st.button("⚠️ Confirm Clear Chat?", type="primary"):
                    reset_chat()
            else:
                reset_chat()


def main():
    """Main function to run the Streamlit app."""
    st.header(
        f"""
    # RAG orchestrated by <img src="data:image/png;base64,{base64.b64encode(open("assets/crewai.png", "rb").read()).decode()}" width="140" style="vertical-align: 0px;">
    """,
        unsafe_allow_html=True,
    )

    display_sidebar()

    # Check if PDF is uploaded before showing chat
    if st.session_state.get("crew") is None:
        st.info("👆 Please upload a PDF document to start chatting")
        return

    # Display chat container
    chat_container = st.container()
    with chat_container:
        # Display previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = st.session_state.crew.kickoff(inputs={"input": prompt})
                        message_placeholder.markdown(result)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result}
                        )
                    except Exception as e:
                        message_placeholder.error(
                            f"Error generating response: {str(e)}"
                        )


if __name__ == "__main__":
    main()
