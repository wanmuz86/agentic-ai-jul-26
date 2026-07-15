from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma # Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splitter
from langchain_community.document_loaders import PyPDFLoader # Load the pdf
from langchain.agents import create_agent

import glob # to open file blob

#retrieve all the in docs folder
pdf_files = glob.glob("./docs/*.pdf")
documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs = loader.load() #load it inside the documents []
    documents.extend(docs)

# chunking the document to be loaded in the DB
# RecursiveCharacterTextSplitter - Is one of the strategy of transforming document into data in Vector DB
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # cut the documents by 1000 characters
    chunk_overlap=150 # With avery chunk, there will be an overlap of 150 characters to preserve the context
)

chunks = splitter.split_documents(documents) # chucks list of texts 

embeddings = OpenAIEmbeddings() # embedding - the process of transforning the text to mathematical representation

# saved inside a vector database
# In vector DB, the word that has the nearest contextual meaning will stored closer to each other
# For each of it will have a number associate to its closeness

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_db"
)

# The retreiever (query) -> that will give at least 3 answers
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Create the rag tools
@tool("rag_search", description="Search PDF knowledge base and return the text.")
def rag_search(query: str) -> str:
    """Search PDF knowledge base and return the text."""
    results = retriever.invoke(query)   # retrieve the data from the documents
    return "\n\n".join([r.page_content for r in results])

tools = [rag_search]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0 # do not be creative , do not hallucinate
)

agent = create_agent(
    model=llm,
    tools=tools,
     system_prompt=(
    "You are an agent to help me with giving information based on knowledge based. "
    "Always use the available tools to answer the given problems and question. "
    "Answer directly with the result, no greetings or chit-chat."
    "If the question asked is not in the document, simply answer 'No information found in KB'" #guardrail
    ),
)


def ask_agent(prompt: str):
    state = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return state["messages"][-1].content

print("Question 1: What are the key steps in our IT incident escalation policy?")
print(ask_agent("What are the key steps in our IT incident escalation policy?"))


print("Question 2: Who is the CEO of Malaysia Airlines?")
print(ask_agent("Who is the CEO of Malaysia Airlines?")) # not inside the document. It will not hallucinate

print("Question 3: Summarize the onboarding SOP in 5 bullet points.")
print(ask_agent("Summarize the onboarding SOP in 5 bullet points.")) # Summarization

print("Question 4: Does the security policy mention daily backups?")
print(ask_agent("Does the security policy mention daily backups?")) # Verify from document




