from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI



tempelate = ChatPromptTemplate.from_messages([
    ("system",
    """ you are an AI that summarizes the text """),
    ("human",
    """ {data} """)
])


model = ChatMistralAI(model="mistral-small-2506")
