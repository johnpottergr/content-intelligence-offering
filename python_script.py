import requests
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
        return [item.get("title", "trend") for item in data]  # Adjust based on ContentStudio API
    except Exception as e:
        print(f"Error fetching trends: {e}", file=sys.stderr)
        return ["trend1", "trend2"]  # Fallback

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2', top_n=10):
    model = SentenceTransformer(model_name)
    keywords = model.encode([text], convert_to_tensor=True)  # For keyword extraction
    embedding = model.encode([text])[0]  # Single embedding
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
        labels, kmeans = cluster_embeddings(np.array(embeddings), n_clusters)  # Keep kmeans for centroids
        centroids = kmeans.cluster_centers_  # Get the average embedding for each cluster
        
        # Identify gap clusters (where my_content has little representation)
        gap_clusters = []
        my_content_labels = labels[len(trends):] if my_content else []
        for cluster_id in range(n_clusters):
            if sum(1 for label in my_content_labels if label == cluster_id) < 1:  # If no my_content in cluster
                gap_clusters.append(cluster_id)
        gap_centroids = [centroids[i] for i in gap_clusters]  # Centroids of gap clusters
        
        # Filter trends with similarity >= 0.5 to any gap centroid
        filtered_texts = []
        filtered_embeddings = []
        for i in range(len(trends)):
            max_similarity = 0.0
            for gap_centroid in gap_centroids:
                similarity = compute_similarity(embeddings[i], gap_centroid)
                max_similarity = max(max_similarity, similarity)
            if max_similarity >= 0.5:  # Keep trend if similar enough to a gap
                filtered_texts.append(texts[i])
                filtered_embeddings.append(embeddings[i])
        
        # Update texts and embeddings with filtered trends and my_content
        texts = filtered_texts + [my_content] if my_content else filtered_texts
        embeddings = filtered_embeddings + [embeddings[-1]] if my_content else filtered_embeddings
        
        # Re-cluster with filtered data if my_content exists and data changed
        if my_content and len(filtered_embeddings) + 1 < len(embeddings):
            labels, _ = cluster_embeddings(np.array(embeddings), n_clusters)
        
        results = []
        for i, (t, emb) in enumerate(zip(texts, embeddings)):
            similarity = compute_similarity(emb, embeddings[-1]) if i < len(embeddings) - 1 else 1.0
            results.append({
                "source": "trend" if i < len(filtered_texts) else "my_content",
                "text": t,
                "cluster": int(labels[i]),
                "similarity_to_my_content": float(similarity)
            })


# Main Execution
def main():
    try:
        input_data = json.loads(sys.stdin.read())
        action = input_data.get("action", "scrape_analyze")
        urls = input_data.get("urls", [])
        my_content = input_data.get("my_content", "")
        n_clusters = input_data.get("n_clusters", 3)

        if action == "scrape_analyze":
            return scrape_analyze(urls, my_content, n_clusters)
        else:
            return json.dumps({"error": "Invalid action"})
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())
