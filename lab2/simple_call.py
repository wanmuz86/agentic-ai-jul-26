from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Slide 3
#Connecting directly to OpenAI 
# 1) Install langchain_openai with pip install
# 2) Setup the API key environment - env

# llm = ChatOpenAI(model="gpt-4o-mini")
# print(llm.invoke("Hello LangChain!").content)

#Slide 8

# def build_chain(llm):
#  # Create a Prompt template and chain it to OpenAI
#  # It can be reusable with specifying different topic
#     return PromptTemplate.from_template("Summarize {topic}") | llm

# chain = build_chain(ChatOpenAI(model="gpt-4o-mini"))

# # THe content is retrieved on .content property
# print(chain.invoke({"topic": "transformers"}))
# print(chain.invoke({"topic": "token in Agentic AI"}))
# print(chain.invoke({"topic": "RAG"}))

# Demo slide 11 : But using CLaude
# Compare this slide 3
# The chane is minimal  lm = ChatOpenAI(model="gpt-4o-mini")
# pip install langchain_anthropic

# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(
#     model="claude-sonnet-5"
# )
# print(llm.invoke("Hello LangChain!").content)

# from langchain_anthropic import ChatAnthropic

# def build_chain(llm):
#     from langchain_core.prompts import PromptTemplate
#     return PromptTemplate.from_template("Summarize {topic}") | llm

 
# def get_llm(provider="openai"):
#     if provider == "openai":
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model="gpt-4o-mini")
#     elif provider == "anthropic":
#         return ChatAnthropic(model="claude-haiku-4-5")
#     raise ValueError("Provider not supported")
# chain = build_chain(get_llm("anthropic"))
# print(chain.invoke({"topic": "prompt engineering"}))

# Slide 13

# llm = ChatOpenAI(model="gpt-4o-mini")
# resp = llm.invoke([
#     SystemMessage(content='You are a friendly support bot. Respond in json as follows {"issue":"", "solution":"","confidence":""}, and specify between 0 to 100'),
#     HumanMessage(content="My Whatsapp freezes when I open settings. I am on iphone 16 pro")
# ])
# print(resp.content)

# # in real use case, we can exract confidence from the resp and do an if-else (rerouting)

# Example passing variable in the  prompt


# prompt = PromptTemplate(
#     input_variables=["issue", "model"],
#     template="Diagnose this issue: {issue} happening on the following device: {model}"
# )
# chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
# print(chain.invoke({"issue": "User device locked out of password","model":"Iphone 16pro"}))


# # Slide 19:
# # Few shot prompting means we provide example on how the response should be stuctured
# # AI will analyze it and format it accordingly

# # Example use case: A re-write/communication agent that will rewrite user's ticket/message into bug report
# fewshot = ChatPromptTemplate.from_messages([
# ("system", "Rewrite bug reports into short Jira titles."), # Role of the agent
# ("human", "app crash when tap profile"),    # Example input
# ("ai", "Crash on profile tap #crash #profile"), # Example output.  < Noun> + which module + action
# ("human", "{bug}") # Input
# ])
# chain = fewshot | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
# print(chain.invoke({"bug": "whatsapp suddenly cannot open"})) 
# #Observe the result and try with more examples
# # my app crash suddenly when press home, why my sound not working on youtube????? / whatsapp suddenly cannot open


#Slide 22


llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
# Define sequential chain steps
step1 = PromptTemplate.from_template("List 5 activities in {city}") | llm | parser
step2 = PromptTemplate.from_template("Make a day plan based on the {acts}") | llm | parser
step3 = PromptTemplate.from_template("Summarize the plan {proposal} in a form of email to client name {name}") | llm | parser
# Run chains sequentially
activities = step1.invoke({"city": "Rome"})
print("Activities:", activities)
itinerary = step2.invoke({"acts": activities})
print("Itinerary:", itinerary)
email = step3.invoke({"proposal":itinerary, "name":"Wan"})
print("Draft email",email)




