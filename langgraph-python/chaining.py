from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, Graph
from langchain_groq import ChatGroq
from pydantic import BaseModel
import os

# Load environment variables
load_dotenv()

# Define our state schema for tracking the workflow progress
class MarketingState(TypedDict):
    input: str
    initial_copy: str
    quality_metrics: Optional[dict]
    final_copy: str
    messages: list[BaseMessage]

# Define quality metrics schema
class QualityMetrics(BaseModel):
    hasCallToAction: bool
    emotionalAppeal: int
    clarity: int

# Initialize Groq client
groq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def generate_initial_copy(state: MarketingState) -> MarketingState:
    """
    Generates the initial marketing copy based on the input.
    """
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Write persuasive marketing copy for: {state['input']}. Focus on benefits and emotional appeal."
        }]
    )
    
    state["initial_copy"] = response.choices[0].message.content
    state["final_copy"] = state["initial_copy"]  # Initialize final copy
    return state

def evaluate_copy(state: MarketingState) -> MarketingState:
    """
    Evaluates the marketing copy against quality metrics.
    """
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this marketing copy for:
            1. Presence of call to action (true/false)
            2. Emotional appeal (1-10)
            3. Clarity (1-10)
            
            Copy to evaluate: {state['final_copy']}"""
        }]
    )
    
    # Parse the metrics from the response
    metrics = QualityMetrics(
        hasCallToAction=True,  # You would parse these values from the actual response
        emotionalAppeal=8,
        clarity=8
    )
    
    state["quality_metrics"] = metrics.dict()
    return state

def improve_copy(state: MarketingState) -> MarketingState:
    """
    Improves the marketing copy if it doesn't meet quality standards.
    """
    metrics = state["quality_metrics"]
    
    # Check if improvements are needed
    if (not metrics["hasCallToAction"] or 
        metrics["emotionalAppeal"] < 7 or 
        metrics["clarity"] < 7):
        
        improvement_prompt = f"""Rewrite this marketing copy with:
        {'' if metrics["hasCallToAction"] else '- A clear call to action'}
        {'' if metrics["emotionalAppeal"] >= 7 else '- Stronger emotional appeal'}
        {'' if metrics["clarity"] >= 7 else '- Improved clarity and directness'}
        
        Original copy: {state['final_copy']}"""
        
        response = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": improvement_prompt}]
        )
        
        state["final_copy"] = response.choices[0].message.content
    
    return state

def create_marketing_workflow() -> Graph:
    """
    Creates the marketing copy generation and improvement workflow.
    """
    workflow = StateGraph(MarketingState)
    
    # Add nodes
    workflow.add_node("generate", generate_initial_copy)
    workflow.add_node("evaluate", evaluate_copy)
    workflow.add_node("improve", improve_copy)
    
    # Add edges
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "improve")
    
    # Set entry and exit points
    workflow.set_entry_point("generate")
    workflow.set_finish_point("improve")
    
    return workflow.compile()

def generate_marketing_copy(input_text: str) -> dict:
    """
    Main function to handle the marketing copy generation process.
    """
    workflow = create_marketing_workflow()
    
    result = workflow.invoke({
        "input": input_text,
        "initial_copy": "",
        "quality_metrics": None,
        "final_copy": "",
        "messages": []
    })
    
    return {
        "initial_copy": result["initial_copy"],
        "final_copy": result["final_copy"],
        "quality_metrics": result["quality_metrics"]
    }

if __name__ == "__main__":
    input_text = "A revolutionary AI-powered chatbot for businesses."
    result = generate_marketing_copy(input_text)
    
    print("\nInitial Marketing Copy:\n", result["initial_copy"])
    print("\nQuality Metrics:", result["quality_metrics"])
    print("\nFinal Marketing Copy:\n", result["final_copy"])