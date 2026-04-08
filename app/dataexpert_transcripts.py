import requests
import json
import os
import uuid
BASE_URL = "http://www.dataexpert.io"  # e.g. https://app.techcreatorzilla.com
API_KEY = os.getenv('OPENAI_API_KEY')    # your student API key
SESSION_ID =  str(uuid.uuid4())  # required by proxy validator
url = f"{BASE_URL}/api/v1/transcript-context"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "x-session-id": SESSION_ID,
}
payload = {
    "query": "What is Iceberg all about when it comes to compaction?"
}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
print(f"Status: {resp.status_code}")
data = resp.json()
print(json.dumps(data, indent=2))
if resp.ok:
    print("\nTop matches:")
    for match in data:
        print(f"\n. Lesson {match['lesson_id']}")
        print(f"   Similarity: {match['similarity']}")
        print(f"   Excerpt: {match['chunk_text']}...")