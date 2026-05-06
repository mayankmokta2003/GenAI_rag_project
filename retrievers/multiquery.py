from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from  langchain_classic.retrievers.multi_query import MultiQueryRetriever


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

llm = ChatMistralAI(model="mistral-small-2506")

emb = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(docs, emb)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

mqr = MultiQueryRetriever.from_llm(retriever=retriever,llm=llm)

result = mqr.invoke("what is gradient descent? ")

for i in result:
    print(i)
    print()


