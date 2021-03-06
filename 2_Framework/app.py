import sys
sys.path.insert(0, './routes')
sys.path.insert(0, './controllers')

from index import controller

from flask import Flask, jsonify, request
# from overtake import get_overtake_blueprint

from urllib.parse import urlparse

import sqlite3

from utils import send_crest_requset, get_crest_data

import threading
from multiprocessing import Process
import redis

import atexit
import datetime
import time

from audioPlayer import audioPlayer

# global Variables
crestThreads = {}
c = controller()

# DB for config
conn = sqlite3.connect("./config/db/test.db")
cur = conn.cursor()

# Getting Simulator info
cur.execute("select * from simulators")
sims = cur.fetchall()
 
# Connection 닫기
conn.close()

r = redis.StrictRedis(host='localhost', port=6379, db=0)

app = Flask(__name__)

def create_app():
    def interrupt():
        global crestThreads
        global listener

        #if crestThread:
        for ip, crestThread in crestThreads.items:
            crestThread.cancel()    

        listener.stop()
        player.stop()
    
    def getPcarsData():
        global crestThreads
        global sims

        for sim in sims:
            crestThread = get_crest_data(sim[0], r)
            crestThreads[sim[0]] = crestThread

        return sims

    def aPlayer(r, sims):
        channels = r.pubsub()
        for sim in sims:
            # 여러 채널이 가능한지 추가 확인 필요
            channels.subscribe('results')

        while True:
            time.sleep(0.1)
            message = r.hgetall('results')
            if message:
                result = {key.decode(): value.decode() for (key, value) in message.items()}
                # 2초보다 오래된 메세지 제거
                msg_time = datetime.datetime.strptime(result['current_time'], '%Y-%m-%d %H:%M:%S.%f')
                ref_time = datetime.datetime.now() - datetime.timedelta(seconds=2)

                if msg_time > ref_time:
                    p = audioPlayer(result)
                    
    getPcarsData()
    
    player = Process(target=aPlayer, args=(r, sims)).start()

    for sim in sims:
        c.checkOvertake(r,sim[0])
        c.chaseChecker(r,sim[0])
        c.lapDistanceChecker(r,sim[0])

    atexit.register(interrupt)
    return app

app = create_app()  

# Add routes
# app.register_blueprint(get_overtake_blueprint)

@app.route('/status', methods=['GET'])
def status():
    global c
    global sims

    for sim in sims:
        c.checkOvertake(r,sim[0])
        c.chaseChecker(r,sim[0])
        c.lapDistanceChecker(r,sim[0])

    return jsonify({}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)