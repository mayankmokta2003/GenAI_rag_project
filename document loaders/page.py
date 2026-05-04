from langchain_community.document_loaders import WebBaseLoader

data = WebBaseLoader("")

docs = data.load()

print(docs[0].page_content)