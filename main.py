from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate



messages = []

llm = ChatMistralAI(model="mistral-small-2506")

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    embedding_function = embedding_model,
    persist_directory = "chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    },
)
# multi_query_retrieval = MultiQueryRetriever.from_llm()



# prompt tempelate


prompt = ChatPromptTemplate.from_messages([
    ("system",
    """ You are an helpful AI assistant
    Use only the provided context to answer the question.
    If the answer is not present in the context,
    say: "I could not find the answer of your question in the document"
       """),
    ("human",
    """ Context: {context} 
    Question: {question} """)
])


print("-----------RAG SYSTEM CREATED----------")
print("Press 0 to exit")








