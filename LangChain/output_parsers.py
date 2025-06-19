from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)

json_parser = JsonOutputParser()

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extraction assistant. Always respond with valid JSON."),
    ("human", """Extract key information from this text and format as JSON:
    
    Text: {text}
    
    Required JSON format:
    {{
        "main_topic": "string",
        "key_points": ["point1", "point2", "point3"],
        "sentiment": "positive/negative/neutral",
        "confidence": 0.95
    }}""")
])

json_chain = extraction_prompt | llm | json_parser


sample_text = "I absolutely love this new smartphone! The camera quality is amazing, the battery lasts all day, and the interface is so intuitive. Definitely recommend it to anyone looking for an upgrade."

result = json_chain.invoke({"text": sample_text})
print(result)


topic = result["main_topic"]
points = result["key_points"]