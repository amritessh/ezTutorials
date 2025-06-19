from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)

outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a story planning assistant."),
    ("human", "Create a brief outline for a {genre} story about {topic}.")
])

story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer."),
    ("human", "Write a complete short story based on this outline:\n{outline}")
])


outline_chain = outline_prompt | llm | StrOutputParser()
story_chain = story_prompt | llm | StrOutputParser()


def create_story(genre, topic):
    outline = outline_chain.invoke({"genre": genre, "topic": topic})
    story = story_chain.invoke({"outline": outline})
    return story


final_story = create_story("horror", "abandoned lighthouse")
print(final_story)