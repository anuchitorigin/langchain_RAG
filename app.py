# To skip the warning "USER_AGENT environment variable not set, consider setting it to identify your requests.",
# add the following line to the top of the file.
# This is about HTTP requests header which Chroma uses to fetch the web page.
# Reference: https://github.com/langchain-ai/rag-from-scratch/issues/24
import os
os.environ['USER_AGENT'] = 'myagent'

################################ INCLUDE ################################
import bs4
from typing_extensions import List, TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, StateGraph, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


################################ DECLARATION ################################
QUESTION1 = "What is Task Decomposition?"
QUESTION2 = "Can you look up some common ways of doing it?"

# -- Define state for application --
class State(TypedDict):
  question: str
  context: List[Document]
  answer: str


################################ FUNCTION ################################
# -- Define application steps --
@tool(response_format="content_and_artifact")
def retrieve(query: str):
  """Retrieve information related to a query."""
  retrieved_docs = vector_store.similarity_search(query, k=2)
  serialized = "\n\n".join(
      (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
      for doc in retrieved_docs
  )
  return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
  """Generate tool call for retrieval or respond."""
  llm_with_tools = llm.bind_tools([retrieve])
  response = llm_with_tools.invoke(state["messages"])
  # MessagesState appends messages to state instead of overwriting
  return {"messages": [response]}


# Step 2: Execute the retrieval.
# tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
  """Generate answer."""
  # - Get generated ToolMessages
  recent_tool_messages = []
  for message in reversed(state["messages"]):
    if message.type == "tool":
      recent_tool_messages.append(message)
    else:
      break
  tool_messages = recent_tool_messages[::-1]
  # - Format into prompt
  docs_content = "\n\n".join(doc.content for doc in tool_messages)
  system_message_content = (
    """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise.
    Cite the source of the information if possible.
    \n\n
    """
    f"{docs_content}"
  )
  conversation_messages = [
    message
    for message in state["messages"]
    if message.type in ("human", "system")
    or (message.type == "ai" and not message.tool_calls)
  ]
  prompt = [SystemMessage(system_message_content)] + conversation_messages
  # - Run
  response = llm.invoke(prompt)
  return {"messages": [response]}


def ask_llm(graph, config, input_message):
  for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
  ):
    step["messages"][-1].pretty_print()
  return

################################ MAIN ################################
print("Loading model...")
# llm = ChatOpenAI(model="gpt-4o-mini", api_key="...")
llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = Chroma(embedding_function=embeddings)

print("Loading blog...")
# -- Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
  bs_kwargs=dict(
    parse_only=bs4.SoupStrainer(
      class_=("post-content", "post-title", "post-header")
    )
  ),
)
docs = loader.load()
# print(docs)
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:500])

print("Splitting data...")
# - This code snippet is for generic text use cases.
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,  # chunk size (characters)
  chunk_overlap=200,  # chunk overlap (characters)
  add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

# -- Index chunks
print("Indexing chunks...")
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])

# -- Compile the LangGraph
print("Compiling LangGraph...")
graph_builder = StateGraph(MessagesState)
tools = ToolNode([retrieve])
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
  "query_or_respond",
  tools_condition,
  {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# -- Run the application
print("Consulting model...")
# > Serial questions
# - Specify an ID for the thread
# config = {"configurable": {"thread_id": "abc123"}}
# ask_llm(graph, config, QUESTION1)
# ask_llm(graph, config, QUESTION2)

# > Agent method
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
# - Specify an ID for the thread
config = {"configurable": {"thread_id": "def234"}}

input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent_executor.stream(
  {"messages": [{"role": "user", "content": input_message}]},
  stream_mode="values",
  config=config,
):
  event["messages"][-1].pretty_print()


################################ END OF MAIN PROGRAM ################################