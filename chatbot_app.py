import streamlit as st
from backend.core import run_llm, get_faiss_vectordb

# Set the title for the Streamlit app.
st.title("RAG Chatbot to query your own files\n:blue[Using FAISS vector database and Mistral-7B Large Language Model]")

# Allow the user to upload a file with supported extensions.
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

# Provide a text input field for the user to ask questions about the uploaded article.
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# If an uploaded file is available, process it.
if uploaded_file:
    # Save the uploaded file locally.
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create a FAISS vector database from the uploaded file.
    vectordb = get_faiss_vectordb(uploaded_file.name)
    
    # If the vector database is not created (unsupported file type), display an error message.
    if vectordb is None:
        st.error(
            f"The {uploaded_file.type} is not supported. Please load a file in pdf, txt, or md"
        )

# Display a spinner while generating a response.
with st.spinner("Generating response..."):
    # If both an uploaded file and a question are available, run the model to get an answer.
    if uploaded_file and question:
        answer = run_llm(vectordb=vectordb, query=question)
        # Display the answer in a Markdown header format.
        st.write("### Answer")
        st.write(f"{answer['result']}")
        # we can add answer['source_documents']
