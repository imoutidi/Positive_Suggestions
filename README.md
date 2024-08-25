### Build images
Make sure you have installed Docker on your system.

From the terminal get into the app folder and run:
```
sudo docker-compose up --build
```

The application will start after the build of the images ends.

To run the application after the build use this:

```
sudo docker-compose up 
```

### You can access the API with python requests

```python
import requests
from requests.auth import HTTPBasicAuth

print(requests.post("http://127.0.0.1:8000/sentence_sentiment",
                    json={"text": "the application was <blank>"},
                    auth=HTTPBasicAuth("user_moutidis", "31415")).json())
```

### Or with a curl script

```
curl -L -X POST "http://127.0.0.1:8000/sentence_sentiment" -H "Content-Type: application/json" -d '{"text": "the application was <blank>"}' -u "user_moutidis:31415"
```

To access the documentation of the API paste on a browser:

> 127.0.0.1:8000/docs

### Locust load testing

After running the server you can access locust and start the testing from
a browser using this address:

```
http://localhost:8089
```

On the host field put
```
http://fastapi-app:8000
```




