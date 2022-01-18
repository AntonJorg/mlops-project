import time
import requests
url = 'https://us-central1-dtumlops-detr.cloudfunctions.net/function-1'
payload = {'message': 'Hello, General Kenobi'}

for _ in range(1000):
   r = requests.get(url, params=payload)

