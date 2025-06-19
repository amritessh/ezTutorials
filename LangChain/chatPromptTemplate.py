from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)

story_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a creative writing assistant. Your stories should be:
    - Engaging and well-structured
    - Appropriate for all audiences
    - Exactly the requested word count"""),
    ("human", "Write a {style} story about {topic} in approximately {word_count} words.")
])

def generate_story(style, topic, word_count):
    formatted_messages = story_prompt.format_messages(
        style=style,
        topic=topic,
        word_count=word_count
    )
    response = llm.invoke(formatted_messages)
    return response.content

story1 = generate_story("mystery", "missing cat", "200")
story2 = generate_story("adventure", "treasure hunt", "300")
story3 = generate_story("romance", "coffee shop", "150")

print(story1)
print(story2)
print(story3)
