from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import subprocess
import json
import os
from openai import OpenAI

app = FastAPI()

class ContentRequest(BaseModel):
    content: str
    viral_content: str = ""  # Will use ContentStudio trends
    insights: str = ""
    hashtags: list[str] = []
    clusters: list[int] = []  # For n8n output
    similarities: list[float] = []  # For n8n output

@app.post("/analyze-viral")
def analyze_viral(request: ContentRequest):
    # Call python_script.py with ContentStudio trends (placeholder API)
    input_data = {
        "action": "scrape_analyze",
        "urls": ["https://api.contentstudio.io/v1/trends?query=AI"],  # Replace with actual ContentStudio API endpoint
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
        "trends": analysis.get("trends", []),  # Adjust based on python_script.py output
        "content_gaps": [f"Cluster {c}" for c in set(request.clusters) if c != 0]
    }


@app.post("/optimize-structure")
def optimize_structure(request: ContentRequest):
    # Get trends and gaps from analyze-viral (assuming called first)
    trends = request.viral_content.split(", ") if request.viral_content else []
    gaps = [f"Cluster {c}" for c in request.clusters if c != 0]
    
    # Craft prompt using trends and gaps
    prompt = f"Optimize the following content: {request.content}. Use these trends: {', '.join(trends)}. Address these content gaps: {', '.join(gaps)}. Provide a 500-word response suitable for a LinkedIn post."
    
    # Initialize DeepSeek client with API key from environment
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
        base_url="https://api.deepseek.com"
    )
    
    # Call DeepSeek API
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # Using DeepSeek-V3-0324
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,  # Approx 500 words
            temperature=1.3   # General conversation setting
        )
        optimized = response.choices[0].message.content.strip()
    except Exception as e:
        optimized = f"Error generating content: {str(e)}"  # Fallback if API fails
    
    return {"optimized_content": optimized}


@app.post("/recommend-hashtags")
def recommend_hashtags(request: ContentRequest):
    return {"hashtags": [f"#{trend.split()[0]}" for trend in request.clusters] or ["#AI", "#Automation", "#Workflows"]}

@app.post("/finalize-content")
def finalize_content(req: ContentRequest):
    hashtags_text = " ".join(req.hashtags)
    final_output = f"{req.content}\n\n{hashtags_text}"
    return {"final_content": final_output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
