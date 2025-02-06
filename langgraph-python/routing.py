from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, Graph
from langchain_groq import ChatGroq
from pydantic import BaseModel
import os

# Load environment variables
load_dotenv()

# Define our state schema
class AgentState(TypedDict):
    query: str
    classification: dict
    response: str
    messages: list[BaseMessage]

# Define classification schema
class QueryClassification(BaseModel):
    reasoning: str
    type: Literal["general", "refund", "technical"]
    complexity: Literal["simple", "complex"]

# Initialize Groq client
groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Step 1: Classify the customer query
def classify_query(state: AgentState) -> AgentState:
    """Classifies the incoming customer query by type and complexity."""
    
    prompt = f"""Classify this customer query:
    {state['query']}

    Determine:
    1. Query type (general, refund, or technical)
    2. Complexity (simple or complex)
    3. Brief reasoning for classification"""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the classification from the response
    classification = QueryClassification(
        reasoning=response.choices[0].message.content,
        type="technical",  # You would parse this from the response
        complexity="simple"  # You would parse this from the response
    )
    
    state["classification"] = classification.dict()
    return state

# Step 2: Generate response based on classification
def generate_response(state: AgentState) -> AgentState:
    """Generates a response based on the query classification."""
    
    # Select model based on complexity
    model = "llama3-8b-8192" if state["classification"]["complexity"] == "simple" else "llama-3.1-8b-instant"
    
    # Select system prompt based on query type
    system_prompts = {
        "general": "You are an expert customer service agent handling general inquiries.",
        "refund": "You are a customer service agent specializing in refund requests. Follow company policies and gather necessary information.",
        "technical": "You are a technical support specialist with in-depth knowledge of the product. Focus on clear, step-by-step troubleshooting."
    }
    
    response = groq.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompts[state["classification"]["type"]]},
            {"role": "user", "content": state["query"]}
        ]
    )
    
    state["response"] = response.choices[0].message.content
    return state

# Create the workflow graph
def create_customer_service_workflow() -> Graph:
    """Creates the customer service workflow graph."""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("respond", generate_response)
    
    # Add edges
    workflow.add_edge("classify", "respond")
    
    # Set entry and exit points
    workflow.set_entry_point("classify")
    workflow.set_finish_point("respond")
    
    return workflow.compile()

# Main execution function
def handle_customer_query(query: str) -> dict:
    """Handles a customer query through the workflow."""
    
    # Initialize workflow
    workflow = create_customer_service_workflow()
    
    # Execute workflow
    result = workflow.invoke({
        "query": query,
        "classification": {},
        "response": "",
        "messages": []
    })
    
    return {
        "classification": result["classification"],
        "response": result["response"]
    }

# Example usage
if __name__ == "__main__":
    query = "I am experiencing a 404 error when trying to log in to my app. How can I fix this?"
    result = handle_customer_query(query)
    print("Classification:", result["classification"])
    print("Response:", result["response"])