import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="RAG AI 🤖", layout="wide")

st.title("📄 RAG PDF Chat App")
st.markdown("Upload PDF → Ask Questions → Get Answers 🔥")

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- LLM ----------------
llm = ChatMistralAI(model="mistral-small-2506")
embedding_model = MistralAIEmbeddings()

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.
Use only the provided context to answer.
If answer not found, say:
"I could not find the answer of your question in the document"
"""),
    ("human",
     """Context: {context}
Question: {question}""")
])

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF Uploaded Successfully")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Create vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    st.session_state.vectorstore = vectorstore
    st.success("✅ Vector DB Created!")

# ---------------- QUERY SECTION ----------------
if st.session_state.vectorstore:
    query = st.text_input("💬 Ask your question")

    if query:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            },
        )

        docs = retriever.invoke(query)

        # context banana
        context = []
        for i in docs:
            context.append(i.page_content)

        context = "\n\n".join(context)

        # final prompt
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.markdown("### 🤖 Answer")
        st.write(response.content)

        # debug section (optional 🔥)
        with st.expander("🔍 Retrieved Context"):
            st.write(context)




#.venv/bin/activate