from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from openai import OpenAI

from langchain_chroma import Chroma
from langchain.agents import create_agent


#Creating the agent

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2, #how strict are you when answerring 0.xx -2 (bigger number tend to hallucinate -> used for creativity)
)

# Creating the memory for the agent

embeddings = OpenAIEmbeddings()

memory_db = Chroma(
    collection_name="agent-memory",
    embedding_function=embeddings,
    persist_directory="./chromadb",
)

# Creating the client
oai_client = OpenAI()

# tools

@tool("web_search", description="Search the web using OpenAI's built-in search tool.")
def web_search(query: str) -> str:
    """Use OpenAI Search to get up-to-date information."""
    
    # Open AI web search tool
    response = oai_client.responses.create(
        model="gpt-4o-mini",
        input=query,
        tools=[{"type": "web_search"}],
        tool_choice={"type": "web_search"},
    )

    # Prefer structured output if available
    if hasattr(response, "output_text"):
        return response.output_text

    return str(response)


@tool("calculator", description="Evaluate a math expression.")
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as text."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error evaluating expression."


@tool("file_writer", description="Write content into output.txt.")
def file_writer(content: str) -> str:
    """Write the given content into output.txt and return a status message."""
    with open("output.txt", "w") as f:  # Writing given content in output.txt file
        f.write(content)
    return "File saved to output.txt"


# Save in temporary memory
@tool("memory_save", description="Store information in long-term vector memory.")
def memory_save(text: str) -> str:
    """Store the given text in the agent's long-term memory."""
    memory_db.add_texts([text])
    return "Memory saved."

# Search data from temporary memory
@tool("memory_search", description="Retrieve information from long-term memory.")
def memory_search(query: str) -> str:
    """Search long-term memory and return the top matching items."""
    results = memory_db.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])



tools = [web_search, calculator, file_writer, memory_save, memory_search]


agent = create_agent(
    model=llm,
    tools=tools,
    # system_prompt=(
    # "You are a miscellenous agent to help me with day to day operation. "
    # "Always use the available tools to solve the given problems. "
    # "Answer directly with the result, no greetings or chit-chat."
    # "If there is no tools available that can resolve the problem, answer 'I don't know'" #guardrail
    # ),
)

def ask_agent(prompt: str) -> str:
    """Send a user message to the agent and return the final answer."""
    print(f"\n---- ASK AGENT ----\n{prompt}\n-------------------")
    state = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    print("---- RAW STATE ----")
    print(state)
    print("-------------------")
    return state["messages"][-1].content


if __name__ == "__main__":
    print(
        ask_agent("Find top 3 AWS security updates today and summarize them.")
    )

    # Test 2: Planning-only task
    print(
        ask_agent("Plan a 3-step workflow to generate a monthly financial report.")
    )


    # Test 3: Save memory
    ask_agent("Remember that our monthly report is due every 25th.")
    
    # Test 4: Recall memory
    print(
        ask_agent("What do you know about our monthly report?")
    )

    # Test 5: File writer
    ask_agent(" Write a summary of what this agent has done so far in a file.")
