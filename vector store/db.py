from dotenv import load_dotenv
load_dotenv()
from langchain_community.retrievers import wik
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.vectorstores import Chroma

from langchain_core.documents import Document


docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings()


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)


result = vectorstore.similarity_search("what is used in deep learning ?", k=2)

for i in result:
    print(i.page_content)
    print()
