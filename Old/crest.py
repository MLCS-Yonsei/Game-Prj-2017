import http.client

conn = http.client.HTTPConnection("165.132.107.250:8080")

headers = {
    'cache-control': "no-cache",
    'postman-token': "ea89763d-b155-4137-4892-7848490c9752"
    }

conn.request("GET", "/crest/v1/api", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))