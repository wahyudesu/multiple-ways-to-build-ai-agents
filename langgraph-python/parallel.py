import os
import json
from langchain_openai import ChatGroq
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, List
from pydantic import BaseModel, Field

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def create_groq_model():
    """
    Initialize the Groq model with the specified API key.
    """
    return ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# Define response schemas using Pydantic
class SecurityReview(BaseModel):
    vulnerabilities: List[str] = Field(default=[])
    risk_level: str = Field(default="low")
    suggestions: List[str] = Field(default=[])

class PerformanceReview(BaseModel):
    issues: List[str] = Field(default=[])
    impact: str = Field(default="low")
    optimizations: List[str] = Field(default=[])

class MaintainabilityReview(BaseModel):
    concerns: List[str] = Field(default=[])
    quality_score: int = Field(default=10)
    recommendations: List[str] = Field(default=[])

# Define a function to perform a review
def review_code(model, system_prompt: str, code: str, response_schema):
    """
    Conducts a specialized review based on the provided system prompt and schema.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Review this code:\n{code}")
    ]
    response = model.invoke(messages)
    return response_schema.parse_raw(response.content)

# Define the workflow graph
class ReviewState(BaseModel):
    code: str
    security: SecurityReview = None
    performance: PerformanceReview = None
    maintainability: MaintainabilityReview = None
    summary: str = ""

graph = StateGraph(ReviewState)

def security_review_node(state: ReviewState):
    model = create_groq_model()
    state.security = review_code(model, "You are an expert in code security...", state.code, SecurityReview)
    return state

def performance_review_node(state: ReviewState):
    model = create_groq_model()
    state.performance = review_code(model, "You are an expert in code performance...", state.code, PerformanceReview)
    return state

def maintainability_review_node(state: ReviewState):
    model = create_groq_model()
    state.maintainability = review_code(model, "You are an expert in code quality...", state.code, MaintainabilityReview)
    return state

def summarize_reviews(state: ReviewState):
    model = create_groq_model()
    messages = [
        SystemMessage(content="You are a technical lead summarizing multiple code reviews."),
        HumanMessage(content=f"Synthesize these code review results:\n{json.dumps(state.dict(), indent=2)}")
    ]
    response = model.invoke(messages)
    state.summary = response.content
    return state

# Add nodes to the graph
graph.add_node("security_review", security_review_node)
graph.add_node("performance_review", performance_review_node)
graph.add_node("maintainability_review", maintainability_review_node)
graph.add_node("summary", summarize_reviews)

# Define parallel execution
graph.add_edge("security_review", "summary")
graph.add_edge("performance_review", "summary")
graph.add_edge("maintainability_review", "summary")

graph.set_entry_point(["security_review", "performance_review", "maintainability_review"])

graph.set_finish_node("summary")

workflow = graph.compile()

def run_parallel_code_review(code: str):
    """
    Runs the full parallel code review workflow.
    """
    initial_state = ReviewState(code=code)
    result = workflow.invoke(initial_state)
    return result
