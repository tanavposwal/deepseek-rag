import re
import os
import tempfile
import base64
import streamlit as st
from ollama import chat

# Set Streamlit page configuration (optional)
st.set_page_config(page_title="Ollama Streaming Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

if "crew" not in st.session_state:
    st.session_state.crew = None


def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )


def display_message(message):
    """Display a single message in the chat interface."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])


def display_assistant_message(content):
    """Display assistant message with thinking content if present."""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        think_content = think_match.group(0)
        response_content = content.replace(think_content, "")
        think_content = format_reasoning_response(think_content)
        with st.expander("Thinking complete!"):
            st.markdown(think_content)
        st.markdown(response_content)
    else:
        st.markdown(content)


def display_chat_history():
    """Display all previous messages in the chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":  # Skip system messages
            display_message(message)


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


def process_thinking_phase(stream):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    with st.status("Thinking...", expanded=False) as status:
        think_placeholder = st.empty()

        for chunk in stream:
            content = chunk["message"]["content"] or ""
            thinking_content += content

            if "<think>" in content:
                continue
            if "</think>" in content:
                content = content.replace("</think>", "")
                status.update(
                    label="Thinking complete!", state="complete", expanded=False
                )
                break
            think_placeholder.markdown(format_reasoning_response(thinking_content))

    return thinking_content


def process_response_phase(stream):
    """Process the response phase of the assistant's response."""
    response_placeholder = st.empty()
    response_content = ""
    for chunk in stream:
        content = chunk["message"]["content"] or ""
        response_content += content
        response_placeholder.markdown(response_content)
    return response_content


@st.cache_resource
def get_chat_model():
    """Get a cached instance of the chat model."""
    return lambda messages: chat(
        model="deepseek-r1",
        messages=messages,
        stream=True,
    )


def handle_user_input():
    """Handle new user input and generate assistant response."""
    if user_input := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            chat_model = get_chat_model()
            stream = chat_model(st.session_state["messages"])

            thinking_content = process_thinking_phase(stream)
            response_content = process_response_phase(stream)

            # Save the complete response
            st.session_state["messages"].append(
                {"role": "assistant", "content": thinking_content + response_content}
            )


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
                    pass
                    # st.session_state.pdf_tool = DocumentSearchTool(
                    #     file_path=temp_file_path
                    # )

            st.success("PDF indexed! Ready to chat.")

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    st.button("Clear Chat", on_click=reset_chat)


def main():
    """Main function to handle the chat interface and streaming responses."""
    st.markdown(
        """
    # Mini ChatGPT powered by <img src="data:image/png;base64,{}" width="140" style="vertical-align: -3px;">
""".format(base64.b64encode(open("assets/deep-seek.png", "rb").read()).decode()),
        unsafe_allow_html=True,
    )

    display_chat_history()
    handle_user_input()


if __name__ == "__main__":
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    main()
