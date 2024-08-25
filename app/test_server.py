import requests
from requests.auth import HTTPBasicAuth

# print(requests.get("http://127.0.0.1:8000/").json())
# # print(requests.get("http://127.0.0.1:8000/items/0").json())
# print(requests.get("http://127.0.0.1:8000/items?name=Hammer&count=20").json())
# print(requests.post("http://127.0.0.1:8000/",
#                     json={"name": "Screwdriver", "price": 3.99, "count": 10, "id": 4, "category": "tools"},
#                     ).json())

# print(requests.post("http://127.0.0.1:8000/sentence_sentiment",
#                     json={"text": "the application was <blank>"},).json())
# print(requests.get("http://127.0.0.1:8000/").json())

# curl -X POST "http://127.0.0.1:8000/sentence_sentiment" -H "Content-Type: text/plain" -d "have a <blank> day" -u user:31415

print(requests.post("http://127.0.0.1:8000/sentence_sentiment",
                    json={"text": "the application was <blank>"},
                    auth=HTTPBasicAuth("user", "31415")).json())