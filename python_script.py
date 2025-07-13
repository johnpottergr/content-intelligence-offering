import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import json
import numpy as np
import os

# Shared Utility Functions
def fetch_contentstudio_trends(url, api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item.get("title", "trend") for item in data]  # Adjust based on ContentStudio API response
    except Exception as e:
        print(f"Error fetching trends: {e}", file=sys.stderr)
        return ["trend1", "trend2"]  # Fallback

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2', top_n=10):
    model = SentenceTransformer(model_name)
    keywords = model.encode([text], convert_to_tensor=True)  # Simplified for embeddings
    embedding = model.encode([text])[0]  # Get single embedding
    return keywords, embedding

def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Content Intelligence Actions
def scrape_analyze(urls, my_content, n_clusters):
    api_key = os.getenv("CONTENTSTUDIO_API_KEY", "your_api_key_here")  # Set in n8n or env
    trends = fetch_contentstudio_trends(urls[0], api_key)
    texts = trends + [my_content] if my_content else trends
    embeddings = [extract_keywords_and_embeddings(t)[1] for t in texts if t]
    if len(embeddings) >= 2:
        n_clusters = min(max(2, len(embeddings) // 2), 10)  # Dynamic clustering
        labels, _ = cluster_embeddings(np.array(embeddings), n_clusters)
        results = []
        for i, (t, emb) in enumerate(zip(texts, embeddings)):
            similarity = compute_similarity(emb, embeddings[-1]) if i < len(embeddings) - 1 else 1.0
            results.append({
                "source": "trend" if i < len(trends) else "my_content",
                "text": t,
                "cluster": int(labels[i]),
                "similarity_to_my_content": float(similarity)
            })
        # Identify gaps: clusters with few my_content points
        gap_clusters = [i for i, label in enumerate(labels[:-1]) if sum(1 for l in labels[len(trends):] if l == label) < 1]
        return json.dumps({"trends": trends, "gaps": [f"Cluster {c}" for c in gap_clusters]})
    return json.dumps({"error": "Insufficient data for clustering"})

# Authority Engine Actions
def generate_content(prompt, api_url="http://localhost:11434/api/generate"):
    try:
        response = requests.post(
            api_url,
            json={"model": "deepseek", "prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return json.dumps({"thread": response.json().get("response", f"Content idea for {prompt}")})
    except Exception as e:
        return json.dumps({"error": f"LLM generation failed: {e}"})

# Main Execution
def main():
    try:
        input_data = json.loads(sys.stdin.read())
        action = input_data.get("action", "scrape_analyze")
        urls = input_data.get("urls", [])
        my_content = input_data.get("my_content", "")
        n_clusters = input_data.get("n_clusters", 3)
        prompt = input_data.get("prompt", "")

        if action == "scrape_analyze":
            return scrape_analyze(urls, my_content, n_clusters)
        elif action == "generate_content":
            return generate_content(prompt)
        else:
            return json.dumps({"error": "Invalid action"})
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())
