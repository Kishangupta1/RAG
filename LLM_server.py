import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()
# Streamlit UI
st.title("Document-based Q&A with RAG")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load document
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # LLM and Embeddings
    llm = HuggingFaceEndpoint(
        repo_id="jdqqjr/DeepSeek-R1-Distill-Llama-3.2-1B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    chat_model = ChatHuggingFace(llm=llm)
    llm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=llm_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # User input
    question = st.text_input("Ask a question about the document")
    if question:
        response = rag_chain.invoke(question)
        st.write("### Answer:")
        st.write(response)
