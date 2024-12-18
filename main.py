# To skip the warning "USER_AGENT environment variable not set, consider setting it to identify your requests.",
# add the following line to the top of the file.
# This is about HTTP requests header which Chroma uses to fetch the web page.
# Reference: https://github.com/langchain-ai/rag-from-scratch/issues/24
import os
os.environ['USER_AGENT'] = 'myagent'

################################ INCLUDE ################################
import bs4
from typing_extensions import List, TypedDict

from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langgraph.graph import START, StateGraph


################################ DECLARATION ################################
QUESTION = "What is Task Decomposition?"

# -- Define state for application --
class State(TypedDict):
  question: str
  context: List[Document]
  answer: str


################################ FUNCTION ################################
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


# -- Define application steps --
def retrieve(state: State):
  retrieved_docs = vector_store.similarity_search(state["question"])
  return {"context": retrieved_docs}


def generate(state: State):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = prompt.invoke({"question": state["question"], "context": docs_content})
  response = llm.invoke(messages)
  return {"answer": response.content}


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
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = text_splitter.split_documents(docs)

# -- Index chunks
print("Indexing chunks...")
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])

# -- Define prompt for question-answering
print("Defining the prompt...")
# prompt = hub.pull("rlm/rag-prompt") # Load the prompt from the hub - don't need it this time
# template = """
#   You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
#   If you don't know the answer, just say that you don't know. <-- This lets the model answer "I don't know" more often.
#   Use three sentences maximum and keep the answer concise. 
#   Question: {question} 
#   Context: {context} 
#   Answer:
# """
# example_messages = prompt.invoke(
#     {"context": "(context goes here)", "question": "(question goes here)"}
# ).to_messages()
# assert len(example_messages) == 1
# print(example_messages[0].content)
template = """
  You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Use three sentences maximum and keep the answer concise.
  Cite the source of the information if possible.
  Question: {question} 
  Context: {context} 
  Answer:
"""
prompt = PromptTemplate.from_template(template)

# -- Retrieve and generate using the relevant snippets of the blog.
print("Consulting model...")
# > Prompt method
retrieved_docs = vector_store.similarity_search(QUESTION)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
prompt = prompt.invoke({"question": QUESTION, "context": docs_content})
answer = llm.invoke(prompt)
print(answer.content)

# > Chain method
# retriever = vector_store.as_retriever()
# chain = (
#   {"context": retriever | format_docs, "question": RunnablePassthrough()}
#   | prompt
#   | llm
#   | StrOutputParser()
# )
# print(chain.invoke(QUESTION))

# > LangGraph: Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()
# response = graph.invoke({"question": QUESTION})
# print(response["answer"])

################################ END OF MAIN PROGRAM ################################