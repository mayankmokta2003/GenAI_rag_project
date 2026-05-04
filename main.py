from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader


model = ChatMistralAI(model="mistral-small-2506")



data = TextLoader("document loaders/notes.txt")
docs = data.load()

tempelate = ChatPromptTemplate.from_messages([
    ("system",
    """ you are an AI that summarizes the text """),
    ("human",
    """ {data} """)
])



prompt = tempelate.format_messages(data = docs[0].page_content)


response = model.invoke(prompt)
print(response.content)