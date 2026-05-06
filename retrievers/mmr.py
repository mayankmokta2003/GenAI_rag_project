from dotenv import load_dotenv
load_dotenv()
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
    docs,emb
)

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

result = similarity_retriever.invoke("what is Gradient descent? ")

print("\n===== Similarity Search Results =====\n")
for i in result:
    print(i)
    print()



mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

mmr_result = mmr_retriever.invoke("what is Gradient descent? ")

print("\n===== Similarity Search Results =====\n")
for j in mmr_result:
    print(j)
    print()




#output: 
    
# ===== Similarity Search Results =====

# page_content='Gradient descent is an optimization algorithm used in machine learning.'
# page_content='Gradient descent is an optimization that minimizes the loss function.'
# page_content='Gradient descent minimizes the loss function.'

# ===== Similarity Search Results =====

# page_content='Gradient descent is an optimization algorithm used in machine learning.'
# page_content='Gradient descent minimizes the loss function.'
# page_content='Neural networks use gradient descent for training.'