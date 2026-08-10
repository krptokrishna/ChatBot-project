from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# -----------------------------
# Tool
# -----------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web and return relevant results.
    """

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        formatted = []

        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            formatted.append(f"{title}\n{body}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def rag_search(query: str) -> str:
    """
    Search uploaded PDF documents from local knowledge base.
    """

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.load_local(
            "faiss_db",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(
            query,
            k=3
        )

        if not docs:
            return "No relevant information found."

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    except Exception as e:
        return f"RAG Error: {str(e)}"
tools = [
    web_search,
    rag_search
]
# -----------------------------
# State
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Nodes
# -----------------------------
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


# -----------------------------
# Graph
# -----------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")

# Finish if no tool call
graph.add_edge("chat_node", END)


# -----------------------------
# Memory
# -----------------------------
memory_context = SqliteSaver.from_conn_string(
    "chatbot.db"
)

memory = memory_context.__enter__()

chatbot = graph.compile(
    checkpointer=memory
)


# -----------------------------
# Response Function
# -----------------------------
def get_response(user_input, thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config
    )

    return result["messages"][-1].content