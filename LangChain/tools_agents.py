from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Create tools using @tool decorator
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations like '25 * 17' or '(100 + 50) / 3'."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # In production, use real search APIs
    return f"Search results for '{query}': [Latest information about {query}]"

@tool
def weather_info(city: str) -> str:
    """Get current weather for a city."""
    # In production, use real weather API
    import random
    temp = random.randint(15, 35)
    condition = random.choice(["sunny", "cloudy", "rainy"])
    return f"Weather in {city}: {temp}°C, {condition}"

# Create agent
tools = [calculator, web_search, weather_info]

def create_agent():
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=5
    )

# Usage
agent = create_agent()

# Multi-step example
result = agent.invoke({
    "input": "Calculate 127 * 83, then search for information about that number"
})
print(result["output"])

# Weather example
weather_result = agent.invoke({
    "input": "What's the weather in Tokyo and calculate 25% of 200"
})
print(weather_result["output"])