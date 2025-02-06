import os
import langgraph
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

ios.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])

class Task(BaseModel):
    purpose: str
    task_name: str
    change_type: Literal["create", "modify", "delete"]

class TaskPlan(BaseModel):
    tasks: List[Task]
    estimated_effort: Literal["low", "medium", "high"]

def generate_task_plan(task_request: str) -> TaskPlan:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a Project Manager responsible for designing an efficient task execution strategy."},
            {"role": "user", "content": f"Create a work plan for the following task: {task_request}"}
        ],
        response_format="json"
    )
    return TaskPlan(**response["choices"][0]["message"]["content"])

def implement_task_change(task: Task):
    worker_prompts = {
        "create": {
            "Audience research": "You are a Business Analyst responsible for audience research.",
            "Content creation": "You are a Content Strategist designing content strategies.",
            "Account management": "You are a Social Media Manager optimizing accounts.",
            "Performance analysis": "You are a Marketing Analyst measuring strategy success."
        },
        "modify": {
            "Account management": "You are a Social Media Manager improving strategies."
        },
        "delete": "You are an Operations Manager identifying unnecessary tasks."
    }

    system_prompt = worker_prompts.get(task.change_type, {}).get(task.task_name, "You are an expert in this field.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Implement changes for the task: {task.task_name}\nPurpose: {task.purpose}\nExplain why and list action items."}
        ],
        response_format="json"
    )
    return response["choices"][0]["message"]["content"]

# Running the task implementation
if __name__ == "__main__":
    task_request = "Develop a social media marketing strategy for a small business"
    task_plan = generate_task_plan(task_request)
    
    task_changes = [
        {
            "task": task,
            "implementation": implement_task_change(task)
        }
        for task in task_plan.tasks
    ]
    
    print("\n===== TASK IMPLEMENTATION COMPLETED =====")
