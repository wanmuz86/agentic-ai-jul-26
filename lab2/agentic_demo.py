from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent # ✅ use create_agent in v1.x
import requests

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



@tool
def weather(city: str) -> str:
    # Normally I specify the input in comment of the tool
    """Get real weather data using API, based on the given city"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
    "q": city,
    "appid": "9fd7a449d055dba26a982a3220f32aa2", # make sure we don't upload public + openAI
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    return data


 # Initialize LLM

llm = ChatOpenAI(model="gpt-4o-mini")


# Create the agent with your own system prompt
agent = create_agent(
    model=llm,
    tools=[add, product, division, weather], # Options of tool to be. used
    system_prompt=(
    "You are a miscellenous agent to help me with day to day operation. "
    "Always use the available tools to solve the given problems. "
    "Answer directly with the result, no greetings or chit-chat."
    "If there is no tools available that can resolve the problem, answer 'I don't know'" #guardrail
    ),
)

# Invoke the agent – new API expects `messages`
result = agent.invoke(
    {
    "messages": [
       
        ("user", "It is raining in Kuala Lumpur now?") 
        ]
    }
)
# `result` has a messages list – last one is the final answer
final_message = result["messages"][-1]
print(final_message.content)
