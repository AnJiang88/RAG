import os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI



def get_faiss_vectordb(file: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)

    # Initiate embeddings using OpenAI.
    # embedding = OpenAIEmbeddings()
    # BGE models on the HuggingFace are the best open-source embedding models, according to LangChain
    # model size 0.13GB, Embedding Dimensions 384, Max Tokens: 512
    embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # or "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

    # Create a unique FAISS index path based on the input file's name.
    faiss_index_path = f"faiss_index_{filename}"

    # Determine the loader based on the file extension.
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        # If the document type is not supported, print a message and return None.
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create a FAISS vector database from the chunked documents and embeddings.
    vectordb = FAISS.from_documents(doc_chunked, embedding)
    
    # Save the FAISS vector database locally using the generated index path.
    vectordb.save_local(faiss_index_path)
    
    # Return the FAISS vector database.
    return vectordb



def run_llm(vectordb, query: str) -> str:
    # Create an instance of the ChatOpenAI with specified settings.
    # openai_llm = ChatOpenAI(temperature=0, verbose=True)

    hf_llm = HuggingFacePipeline.from_model_id(
                model_id="gpt2",  # mistralai/Mistral-7B-v0.1
                task="text-generation",
                pipeline_kwargs={"temperature": 0, "max_new_tokens": 300}
            )
    
    # Create a RetrievalQA instance from a chain type with a specified retriever.
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=hf_llm, chain_type="stuff", retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    )
    
    # Run a query using the RetrievalQA instance.
    answer = retrieval_qa.invoke({"query": query})
    
    # Return the answer obtained from the query.
    return answer