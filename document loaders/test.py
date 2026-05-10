## split by characters

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter


# splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size=10,
#     chunk_overlap=1
# )

# data = TextLoader("document loaders/notes.txt")

# docs = data.load()

# chunks = splitter.split_documents(docs)

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()





## token based splitting (tiktoken)

    
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# data = PyPDFLoader("document loaders/GRU.pdf")

# docs = data.load()

# splitter = TokenTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10
# )

# chunks = splitter.split_documents(docs)

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()




# recursive charater splitting


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/deeplearning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(docs)
