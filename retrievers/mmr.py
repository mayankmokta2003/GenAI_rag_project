from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

emb = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=emb
)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity"
    search_kwargs={"k":2}
)


