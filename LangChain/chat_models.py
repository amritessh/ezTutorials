from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)


response = llm.invoke("Explain what is a transformer in LLMs?")
print(response.content)

messages = [
    SystemMessage(content="You explain complex topics in simple terms."),
    HumanMessage(content="What is Transformer in LLMs?")
]

response = llm.invoke(messages)
print(response.content)