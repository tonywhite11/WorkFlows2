import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Ensure OPENAI_API_KEY is set in your environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://work-flows2-tony.vercel.app/"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

from typing import Optional

class WorkflowRequest(BaseModel):
    goal: str
    role: str
    region: Optional[str] = None

class CoachChatRequest(BaseModel):
    user_message: str
    step_context: str # Context now includes Goal, Current Step details

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
      "workflowTitle": "A concise and engaging title for the workflow based on the user's goal (e.g., 'Launching Your Etsy Store')",
      "summary": "A brief overall summary of the workflow in 1-2 sentences, highlighting the main phases or outcome.",
      "steps": [
        {{
          "step_number": "integer, starting from 1",
          "action": "string - Describe the main action/task for this step. Be specific and actionable. Use Markdown for emphasis if needed (e.g., **bold** for key terms).",
          "tools": [ 
            {{ "name": "string - Name of the tool or resource", "url": "string - HTTPS URL (official, safe, and working) or null if not applicable (e.g. 'General Research')" }}
          ],
          "time_estimate": "string - e.g., '2-4 hours', '1 day', '1 week'",
          "estimated_cost": "string - e.g., 'USD 0', 'USD 20-50 (software subscription)', 'Varies'. Specify currency (USD if region not specified or cost is global).",
          "alternative": "string - One concise alternative method, tool, or approach for this step. Use Markdown for emphasis if needed."
        }}
      ],
      "total_estimated_cost": "string - A summary of the total estimated cost for the entire project (e.g., 'USD 50-100 for initial setup')."
    }}

    Important rules for the JSON content:
    - Ensure the entire response is a single, valid JSON object.
    - For "tools", provide 1-3 relevant tools per step. If a tool is a general concept (e.g., 'Market Research'), its URL can be null or link to an authoritative guide. All URLs must be HTTPS.
    - "estimated_cost" and "total_estimated_cost" should be in USD unless the region strongly suggests another currency and it's a local cost.
    - The number of steps should be appropriate for the goal, typically 3-7 high-level steps. Each step should be a significant phase.
    - All string values within the JSON must be properly escaped (e.g., double quotes inside strings should be \\").
    - Do not use comments inside the JSON.
    """
    response_content = None
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using the specified model
            messages=[
                {"role": "system", "content": "You are an expert project planner. Your output MUST be a single, valid JSON object as per the user's specified structure. Do not add any explanatory text outside the JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} # Request JSON mode if supported by model version and client
        )
        response_content = response.choices[0].message.content
        if response_content is None:
            raise HTTPException(status_code=500, detail="AI returned an empty response.")
        response_content = response_content.strip()
        
        # Validate and parse the JSON
        workflow_data = json.loads(response_content)

        # Basic validation of the structure
        if not isinstance(workflow_data, dict) or "steps" not in workflow_data or not isinstance(workflow_data["steps"], list):
            raise ValueError("Invalid root structure or 'steps' array missing/invalid.")
        if not workflow_data.get("workflowTitle") or not workflow_data.get("total_estimated_cost"):
             raise ValueError("Missing 'workflowTitle' or 'total_estimated_cost'.")
        for step in workflow_data["steps"]:
            if not isinstance(step, dict) or not step.get("action"):
                raise ValueError("Invalid step structure or 'action' missing in a step.")
            if "tools" in step and not isinstance(step["tools"], list):
                 raise ValueError("Invalid 'tools' format in a step; expected a list.")
            if "tools" in step:
                for tool_item in step["tools"]:
                    if not isinstance(tool_item, dict) or "name" not in tool_item:
                        raise ValueError("Invalid tool item structure; 'name' is required.")
        
        return workflow_data

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Problematic AI response: {response_content if 'response_content' in locals() else 'Response not available'}")
        raise HTTPException(status_code=500, detail={"error": "AI returned invalid JSON format.", "raw_response": response_content  if 'response_content' in locals() else 'Response not available'})
    except ValueError as e: # For custom validation
        print(f"ValueError validating JSON structure: {e}")
        print(f"Problematic AI response: {response_content if 'response_content' in locals() else 'Response not available'}")
        raise HTTPException(status_code=500, detail={"error": f"AI JSON structure error: {e}", "raw_response": response_content if 'response_content' in locals() else 'Response not available'})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the workflow.")


@app.post("/coach-chat")
async def coach_chat(data: CoachChatRequest):
    prompt = (
        "You are a helpful, expert project coach. The user is working on a project and has provided the following context about their current step:\n"
        f"```\n{data.step_context}\n```\n\n" # step_context now includes Goal and full step details
        f"The user's question or statement is: \"{data.user_message}\"\n\n"
        "Please provide a clear, actionable, and supportive response. Your advice should be concise and directly address the user's query. "
        "If the user seems stuck with a tool mentioned in their context, offer specific guidance or suggest alternatives if appropriate. "
        "Use Markdown for formatting if it enhances readability (e.g., bullet points, bold text)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using the specified model
            messages=[
                {"role": "system", "content": "You are a supportive and expert workflow coach. Help users overcome challenges in their project steps."},
                {"role": "user", "content": prompt}
            ]
        )
        message_content = response.choices[0].message.content
        if message_content is None:
            raise HTTPException(status_code=500, detail="AI returned an empty response.")
        reply = message_content.strip()
        return {"reply": reply}
    except Exception as e:
        print(f"Error in coach-chat: {e}")
        raise HTTPException(status_code=500, detail="Error contacting AI coach.")