from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ContentRequest(BaseModel):
    content: str
    viral_content: str = ""
    insights: str = ""
    hashtags: list[str] = []

@app.post("/analyze-viral")
def analyze_viral(request: ContentRequest):
    return {"trends": ["AI automation", "workflow optimization"], "content_gaps": ["automation workflows"]}

@app.post("/optimize-structure")
def optimize_structure(request: ContentRequest):
    return {"optimized_content": request.content + " Include trends: " + ", ".join(["AI automation workflows"])}

@app.post("/recommend-hashtags")
def recommend_hashtags(request: ContentRequest):
    return {"hashtags": ["#AI", "#Automation", "#Workflows"]}

@app.post("/finalize-content")
def finalize_content(req: ContentRequest):
    hashtags_text = " ".join(req.hashtags)
    final_output = f"{req.content}\n\n{hashtags_text}"
    return {"final_content": final_output}
