import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from IPython.display import  display, Image
from langgraph.prebuilt import  ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
model = ChatOpenAI(model='gpt-4o')

class State(MessagesState) :
    summary : str
    
def call_model(state : State) -> State :
    summary = state.get("summary", "")
    
    if summary :
            system_message = f"summary of the conversation earlier : {summary} "
            messages = [SystemMessage(content=system_message)] + state['messages']
            
    else : 
            messages = state['messages'] 
            response = model.invoke(messages)
            return{"messages" : response }           
        
def summarize_conversation(state : State) :
    summary = state.get("summary", "")
    
    if summary : 
                f"This is a summary of conversation to date : {summary} \n\n"
                "Extend the summary by taking into account the new message above"
    else :
        summary_message = f"create a summary of above conversation"
        
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id)for m in state["messages"][:-2]]
    return {"summary" : response.content, "messages" : delete_messages}                

def should_continue(state : State) :
    """Return the next node to execute."""
    messages = state['messages']
    
    if len(messages) > 6:
        return "summarize_conversation"

    return END

builder : StateGraph = StateGraph(MessagesState)

builder.add_node("conversation", call_model)
builder.add_node(summarize_conversation)

builder.add_edge(START, "conversation")
builder.add_conditional_edges("conversation", should_continue)
builder.add_edge('summarize_conversation', END)

memory : MemorySaver = MemorySaver()

graph : CompiledStateGraph = builder.compile(checkpointer=memory)

display(Image(graph.get_graph().draw_mermaid_png()))            