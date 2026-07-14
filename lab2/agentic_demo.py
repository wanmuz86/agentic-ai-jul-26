from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent # ✅ use create_agent in v1.x

# Define a tool
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""  #Comment to describe the tool to the agent , it will use this when selecting the tool
    return a + b

# To add a new 
# 1) You use decorator @ tool
# 2) Add on the list of tools ...
@tool
def product(a:float, b:float)-> float:
    """Perform multiplication operation on two numbers.""" 
    return a * b 

@tool
def division(a:float, b:float)-> float:
    """Perform division operation on two numbers"""
    return a / b

 # Initialize LLM

llm = ChatOpenAI(model="gpt-4o-mini")


# Create the agent with your own system prompt
agent = create_agent(
    model=llm,
    tools=[add, product, division], # Options of tool to be. used
    system_prompt=(
    "You are a precise math agent. "
    "Always use the available tools for calculations. "
    "Answer directly with the result, no greetings or chit-chat."
    "If there is no tools available that can resolve the problem, answer 'I don't know'" #guardrail
    ),
)
# Invoke the agent – new API expects `messages`
result = agent.invoke(
    {
    "messages": [
       
        ("user", "Use your tools to calculate 10 divide by 3") # Some agent will answer it as 3.5 + (-4.5)
        ]
    }
)
# `result` has a messages list – last one is the final answer
final_message = result["messages"][-1]
print(final_message.content)
