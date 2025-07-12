#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Shared Utility Functions
def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=' ')
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error fetching URL: {e}", file=sys.stderr)
        return ""

def extract_keywords_and_embeddings(text, model_name='all-MiniLM-L6-v2', top_n=10):
    model = SentenceTransformer(model_name)
    kw_model = KeyBERT(model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    embedding = model.encode([text])[0]
    return keywords, embedding

def cluster_embeddings(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

def find_optimal_clusters(embeddings, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('elbow_plot.png')
    return inertias.index(min(inertias[1:])) + 2

def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Content Intelligence Actions
def scrape_analyze(urls, my_content, n_clusters):
    texts = [extract_text_from_url(url) for url in urls if url]
    if my_content:
        texts.append(my_content)
    embeddings = [extract_keywords_and_embeddings(t)[1] for t in texts if t]
    if len(embeddings) >= 2:
        optimal_n = find_optimal_clusters(np.array(embeddings))
        n_clusters = min(max(2, optimal_n), len(embeddings))
        labels, _ = cluster_embeddings(np.array(embeddings), n_clusters)
        results = []
        for i, (t, emb) in enumerate(zip(texts, embeddings)):
            keywords, _ = extract_keywords_and_embeddings(t)
            similarity = compute_similarity(emb, embeddings[-1]) if i < len(embeddings) - 1 else 1.0
            results.append({
                "url": urls[i] if i < len(urls) else "my_content",
                "text": t,
                "keywords": keywords,
                "cluster": int(labels[i]),
                "similarity_to_my_content": float(similarity)
            })
        return json.dumps(results)
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
        return json.dumps({"thread": response.json().get("response", "Generated content")})
    except Exception as e:
        return json.dumps({"error": f"LLM generation failed: {e}"})

def post_reply(post_id, content, access_token="YOUR_X_ACCESS_TOKEN"):
    try:
        response = requests.post(
            "https://api.x.com/2/tweets",
            json={"reply": {"in_reply_to_tweet_id": post_id, "text": content}},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return json.dumps({"status": "posted"})
    except Exception as e:
        return json.dumps({"error": f"Post failed: {e}"})

# Main Execution
def main():
    try:
        input_data = json.loads(sys.stdin.read())
        action = input_data.get("action", "scrape_analyze")
        urls = input_data.get("urls", [])
        my_content = input_data.get("my_content", "")
        n_clusters = input_data.get("n_clusters", 3)
        prompt = input_data.get("prompt", "")
        post_id = input_data.get("post_id", "")
        content = input_data.get("content", "")

        if action == "scrape_analyze":
            return scrape_analyze(urls, my_content, n_clusters)
        elif action == "generate_content":
            return generate_content(prompt)
        elif action == "post_reply":
            return post_reply(post_id, content)
        else:
            return json.dumps({"error": "Invalid action"})
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    print(main())
