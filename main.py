from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import subprocess
import json
import os

app = FastAPI()

class ContentRequest(BaseModel):
    content: str
    viral_content: str = ""
    insights: str = ""
    hashtags: list[str] = []
    clusters: list[int] = []  # New field for n8n output
    similarities: list[float] = []  # New field for n8n output

@app.post("/analyze-viral")
def analyze_viral(request: ContentRequest):
    # Call python_script.py to analyze with scrape_analyze
    input_data = {
        "action": "scrape_analyze",
        "urls": ["https://api.x.com/2/tweets/search/recent?query=AI%20automation"],  # Placeholder, replace with dynamic URLs
        "my_content": request.content,
        "n_clusters": 3
    }
    result = subprocess.run(
        ["python3", "python_script.py"],
        input=json.dumps(input_data),
        text=True,
        capture_output=True
    )
    analysis = json.loads(result.stdout) if result.returncode == 0 else {"error": "Analysis failed"}
    return {
        "trends": [item["text"] for item in analysis if isinstance(analysis, list)],
        "content_gaps": [f"Cluster {c}" for c in set(request.clusters) if c != 0]
    }

@app.post("/optimize-structure")
def optimize_structure(request: ContentRequest):
    prompt = f"Optimize {request.content} with trends {', '.join(request.clusters)} and insights {request.insights}"
    input_data = {"action": "generate_content", "prompt": prompt}
    result = subprocess.run(
        ["python3", "python_script.py"],
        input=json.dumps(input_data),
        text=True,
        capture_output=True
    )
    optimized = json.loads(result.stdout).get("thread", request.content) if result.returncode == 0 else request.content
    return {"optimized_content": optimized}

@app.post("/recommend-hashtags")
def recommend_hashtags(request: ContentRequest):
    # Simple logic based on clusters, enhance with LLM if needed
    return {"hashtags": [f"#{trend.split()[0]}" for trend in request.clusters] or ["#AI", "#Automation", "#Workflows"]}

@app.post("/finalize-content")
def finalize_content(req: ContentRequest):
    hashtags_text = " ".join(req.hashtags)
    final_output = f"{req.content}\n\n{hashtags_text}"
    return {"final_content": final_output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
