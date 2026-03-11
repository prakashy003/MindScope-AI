import urllib.request as urllib
import json

data = json.dumps({"text": "I feel sad"}).encode("utf-8")
req = urllib.Request("http://127.0.0.1:5001/predict", data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.urlopen(req) as response:
        print(response.read().decode("utf-8"))
except Exception as e:
    print("Error:", e)
