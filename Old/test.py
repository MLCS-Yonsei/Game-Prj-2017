from rq import Queue
from worker import conn
import http.client
from time import sleep
from utils import send_crest_requset, RepeatedTimer, scan_port

ip = '192.168.0.2'
q = Queue(connection=conn)
def main():
    result = q.enqueue(send_crest_requset, ip + ':8080')

print("starting...")
print(scan_port(ip,8080))
rt = RepeatedTimer(0.1, main) # it auto-starts, no need of rt.start()
try:
    conn = http.client.HTTPConnection('192.168.0.2:8080')
    conn.request("GET", "/crest/v1/api")

    res = conn.getresponse()
    data = res.read()

    print(data)
    sleep(3) # your long-running job goes here...
finally:
    rt.stop() # better in a try/finally block to make sure the program ends!
# import redis
# import time
# r = redis.StrictRedis(host='localhost', port=6379, db=0)
# p = r.pubsub()
# p.subscribe('message')

# while True:
#     message = p.get_message()
#     if message:
#         print "Subscriber: %s" % message['data']
#     time.sleep(1)