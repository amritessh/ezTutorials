from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

system_msg = SystemMessage(content="You are a helpful coding assistant.")

human_msg = HumanMessage(content="Explain Python decorators")

ai_msg = AIMessage(content="Decorators are functions that modify other functions...")