import requests
from requests.auth import HTTPBasicAuth

print(requests.post("http://127.0.0.1:8000/sentence_sentiment",
                    json={"text": "the application was <blank>"},
                    auth=HTTPBasicAuth("user_moutidis", "31415")).json())