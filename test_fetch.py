import urllib.request
import json

def fetch(url, key=None):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode())
            print("Models:", [m["id"] for m in data.get("data", [])])
    except Exception as e:
        print("Error:", e)

# Test with a public or dummy endpoint if possible. We can't really hit openai without a key.
print("ok")
