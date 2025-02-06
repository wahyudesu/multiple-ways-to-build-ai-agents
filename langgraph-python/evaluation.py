from langgraph import LangGraph, Node, Edge
from langgraph.nodes import GenerateTextNode, GenerateObjectNode
from langgraph.schema import TextPrompt, ObjectSchema
from typing import Dict, Any, List
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq API
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the LangGraph
graph = LangGraph()

# Define nodes for the graph

# Node 1: Generate initial article
def generate_initial_article(input_data: Dict[str, Any]) -> Dict[str, Any]:
    model = "llama-3.1-8b-instant"
    system = "You are a writer. Your task is to write a concise article in only 6 sentences! You might get additional feedback from your supervisor!"
    prompt = TextPrompt(f"Write a 6-sentence article on the topic: {input_data['topic']}")
    generate_text_node = GenerateTextNode(model=model, system=system, prompt=prompt)
    result = generate_text_node.execute()
    return {"current_article": result.text, "iterations": 0}

node1 = Node(name="generate_initial_article", function=generate_initial_article)

# Node 2: Evaluate article quality
def evaluate_article_quality(input_data: Dict[str, Any]) -> Dict[str, Any]:
    model = "llama-3.3-70b-versatile"
    schema = ObjectSchema({
        "qualityScore": (int, 1, 10),
        "clearAndConcise": bool,
        "engaging": bool,
        "informative": bool,
        "specificIssues": List[str],
        "improvementSuggestions": List[str]
    })
    system = "You are a writing supervisor! Your agency specializes in concise articles! Your task is to evaluate the given article and provide feedback for improvements! Repeat until the article meets your requirements!"
    prompt = TextPrompt(f"Evaluate this article:\n\nArticle: {input_data['current_article']}\n\nConsider:\n1. Overall quality\n2. Clarity and conciseness\n3. Engagement level\n4. Informative value")
    generate_object_node = GenerateObjectNode(model=model, schema=schema, system=system, prompt=prompt)
    result = generate_object_node.execute()
    return {"evaluation": result.object}

node2 = Node(name="evaluate_article_quality", function=evaluate_article_quality)

# Node 3: Check if quality meets threshold
def check_quality_threshold(input_data: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = input_data["evaluation"]
    if (evaluation["qualityScore"] >= 8 and
        evaluation["clearAndConcise"] and
        evaluation["engaging"] and
        evaluation["informative"]):
        return {"meets_threshold": True}
    else:
        return {"meets_threshold": False}

node3 = Node(name="check_quality_threshold", function=check_quality_threshold)

# Node 4: Generate improved article based on feedback
def improve_article(input_data: Dict[str, Any]) -> Dict[str, Any]:
    model = "llama-3.3-70b-versatile"
    evaluation = input_data["evaluation"]
    system = "You are an expert article writer."
    prompt = TextPrompt(f"Improve this article based on the following feedback:\n{evaluation['specificIssues']}\n{evaluation['improvementSuggestions']}\n\nCurrent Article: {input_data['current_article']}")
    generate_text_node = GenerateTextNode(model=model, system=system, prompt=prompt)
    result = generate_text_node.execute()
    return {"current_article": result.text, "iterations": input_data["iterations"] + 1}

node4 = Node(name="improve_article", function=improve_article)

# Node 5: Final output
def final_output(input_data: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Final Article:\n{input_data['current_article']}")
    print(f"Iterations Required: {input_data['iterations']}")
    return {}

node5 = Node(name="final_output", function=final_output)

# Define edges between nodes
edge1 = Edge(source=node1, target=node2)
edge2 = Edge(source=node2, target=node3)
edge3 = Edge(source=node3, target=node4, condition=lambda x: not x["meets_threshold"])
edge4 = Edge(source=node3, target=node5, condition=lambda x: x["meets_threshold"])
edge5 = Edge(source=node4, target=node2)

# Add nodes and edges to the graph
graph.add_node(node1)
graph.add_node(node2)
graph.add_node(node3)
graph.add_node(node4)
graph.add_node(node5)
graph.add_edge(edge1)
graph.add_edge(edge2)
graph.add_edge(edge3)
graph.add_edge(edge4)
graph.add_edge(edge5)

# Define the input data
input_data = {"topic": "Machine learning in agriculture"}

# Execute the graph
graph.execute(input_data)