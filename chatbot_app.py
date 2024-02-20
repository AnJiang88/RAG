import streamlit as st
from backend.core import run_llm, get_faiss_vectordb

# Set the title for the Streamlit app.
st.title("RAG Chatbot to query your own files\n:blue[Using GPT-3.5, BGE embedding model and FAISS vectoreDB]")

# Allow the user to provide their OpenAI API key.
API_Key = st.sidebar.text_input("First, enter your OpenAI API key", type="password")

# Allow the user to upload a file with supported extensions.
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

# Provide a text input field for the user to ask questions about the uploaded article.
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file or not API_Key,
)

# If both API key and an uploaded file is available, process it.
if API_Key and uploaded_file:
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
    # If an API key, an uploaded file and a question are all available, run the model to get an answer.
    if API_Key and uploaded_file and question:
        answer = run_llm(api_key=API_Key, vectordb=vectordb, query=question)
        # Display the answer in a Markdown header format.
        st.write("### Answer")
        st.write(f"{answer['result']}")
        st.write("### Relevant source")
        rel_docs = answer['source_documents']
        for i, doc in enumerate(rel_docs):
            st.write(f"**{i+1}**: {doc.page_content}\n")
