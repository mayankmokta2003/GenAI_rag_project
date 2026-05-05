from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = ChatMistralAI(model="mistral-small-2506")

data = TextLoader("document loaders/deeplearning.txt")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunk = splitter.split_documents(docs)

tempelate = ChatPromptTemplate.from_messages([
    ("system",
    """ you are an AI that summarizes the text """),
    ("human",
    """ {data} """)
])



prompt = tempelate.format_messages(data = docs[0].page_content)


response = model.invoke(prompt)
print(response.content)