from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()
memory = MemorySaver()
chatbot = graph.compile(
    checkpointer=memory
)
config = {
    "configurable": {
        "thread_id": "krishna-chat"
    }
}

print("🤖 ChatBot Started!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Bye! 👋")
        break

    state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    result = chatbot.invoke(state,config=config)

    print("Bot:", result["messages"][-1].content)