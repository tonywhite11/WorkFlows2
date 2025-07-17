import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

# === Initialize App ===
app = FastAPI()

# === CORS: Allow frontend from Vercel ===
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "https://work-flows2.vercel.app",
    "https://workflows2.onrender.com",
    "http://localhost:3000"
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# === Debug Ping Route ===
@app.get("/ping")
def ping():
    return {"status": "backend alive"}

# === Load API Key ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

# === Role Prompt Data ===
roles_prompt = {
    "Generalist": "You are a versatile individual capable of handling various tasks and adapting to different project needs.",
    "Developer": "You are a senior software engineer and technical project lead.",
    "Designer": "You are a creative UI/UX designer with a focus on user-centric solutions.",
    "Marketer": "You are a strategic digital marketing expert.",
    "Coach": "You are an experienced productivity and project coach.",
    "Entrepreneur": "You are a seasoned startup founder and innovative business strategist.",
    "Student": "You are an academic advisor and study skills expert helping students plan their projects.",
    "Researcher": "You are a meticulous scientific researcher and project planner.",
    "Content Creator": "You are a professional content creator and social media strategist.",
    "Event Planner": "You are an experienced and highly organized event planner.",
    "HR Specialist": "You are a human resources specialist focusing on project-based tasks and onboarding.",
    "Teacher": "You are an experienced teacher and curriculum designer.",
    "Consultant": "You are a pragmatic business consultant focused on actionable strategies.",
    "Healthcare Professional": "You are a healthcare project manager or administrator.",
    "Engineer": "You are a project engineer with a focus on practical implementation.",
    "Writer": "You are a professional writer, editor, and content strategist.",
    "Salesperson": "You are a sales strategist and account planner.",
    "Financial Advisor": "You are a financial advisor helping clients plan financial goals."
}

# === Models ===
class WorkflowRequest(BaseModel):
    goal: str
    role: str
    region: Optional[str] = None

class CoachChatRequest(BaseModel):
    user_message: str
    step_context: str

# === Workflow Route ===
@app.post("/generate-workflow")
async def generate_workflow(data: WorkflowRequest):
    region_str = f" The user is located in {data.region} and this might affect tool availability or costs." if data.region else ""
    
    prompt = f"""
    {roles_prompt.get(data.role, 'You are a helpful planning assistant.')}
    A user wants to achieve the following goal: "{data.goal}".
    {region_str}

    Break this goal down into a clear, actionable workflow.
    Please respond ONLY with a single JSON object. Do not include any text before or after the JSON object.
    The JSON object must have the following structure:
    {{
      "workflowTitle": "A concise and engaging title for the workflow",
      "summary": "A brief summary of the workflow",
      "steps": [
        {{
          "step_number": "integer",
          "action": "string",
          "tools": [{{ "name": "string", "url": "string or null" }}],
          "time_estimate": "string",
          "estimated_cost": "string",
          "alternative": "string"
        }}
      ],
      "total_estimated_cost": "string"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert project planner. Output only a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        if not response_content:
            raise HTTPException(status_code=500, detail="Empty response from AI.")
        
        return json.loads(response_content.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow generation failed: {e}")

# === Coach Chat Route ===
@app.post("/coach-chat")
async def coach_chat(data: CoachChatRequest):
    try:
        prompt = (
            "You are a helpful, expert project coach. The user is working on a project step:\n"
            f"{data.step_context}\n\n"
            f"Their message: \"{data.user_message}\"\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're a smart and supportive coach."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        if content is None:
            raise HTTPException(status_code=500, detail="Empty response from AI.")
        return {"reply": content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach chat failed: {e}")
