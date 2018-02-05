import threading
import atexit
import redis
import datetime

import json

r = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = r.pubsub()

from flask import Flask, request, send_from_directory

from utils import send_crest_requset, RepeatedTimer, scan_port
from videoRecorder import VideoRecorder

POOL_TIME = 0.1 #Seconds

# variables that are accessible from anywhere
commonDataStruct = {}
# lock to control access to variable
dataLock = threading.Lock()
# thread handler

app = Flask(__name__, static_url_path='/video')

def create_app():
    def interrupt():
        global crestThread
        crestThread.cancel()    
        vr.stop()

        message = "Stopping dataCollector.py"
        r.publish('message', message)
    # def doStuff():
    #     # 실행시 돌아가는 코드

    #def doStuffStart():
        # print("Start scanning ports..")
        # for i in range(2,255):
        #     url = "192.168.0." + str(i)
        #     print(scan_port(url,8080))


        # Do initialisation stuff here
        # global crestThread
        # # Create your thread
        # crestThread = threading.Timer(POOL_TIME, doStuff, ())
        # crestThread.start()

    # Initiate
    # doStuffStart()
    # When you kill Flask (SIGTERM), clear the trigger for the next thread


    atexit.register(interrupt)
    return app

app = create_app()  

recording = False
collecting = False

camera_number = 3
resolution = "1280x720"

current_time = datetime.datetime.now()

vr = VideoRecorder(camera_number, resolution, current_time)

def record_start():
    global recording
    vr.start()
    recording = True
    message = "Recording Started!"
    r.publish('message', message)

def record_stop():
    global recording
    vr.stop()
    recording = False
    message = "Recording Stopped!"
    r.publish('message', message)

@app.route('/signal/record', methods=['GET'])
def get_record_signal():
    global recording
    
    if request.args.get('record') == "True" and recording == False:
        record_start()
    elif request.args.get('record') == "False" and recording == True:
        record_stop()

    return 'Signal Received'

def getCrestData(target_ip):
    global commonDataStruct
    global crestThread
    global recording

    with dataLock:
    # 데이터 가져오기
        crest_data = send_crest_requset(target_ip, "crest-monitor", {})
        
        gameState = crest_data['gameStates']['mGameState']

        r.publish('gameState', gameState)

        if gameState > 1:
        # 게임 플레이중
            current_time = str(datetime.datetime.now())
            gamedata = [current_time, crest_data]

            r.publish('message', gamedata)

            if recording == False:
                record_start()
        else:
        # 플레이 종료
            if recording == True:
                record_stop()
    
    # Set the next thread to happen
    crestThread = threading.Timer(POOL_TIME, getCrestData, [target_ip])
    crestThread.start()   

@app.route('/signal/gamedata', methods=['GET'])
def get_data_signal():
    global collecting
    global crestThread
    if request.args.get('collect') == "True" and collecting == False:
        target_ip = "192.168.0.2:9090"

        crestThread = threading.Timer(POOL_TIME, getCrestData, [target_ip])
        crestThread.start()

        collecting = True

        message = "Collecting Started!"
        r.publish('message', message)
    elif request.args.get('collect') == "False" and collecting == True:
        crestThread.cancel()
        
        collecting = False

        message = "Collecting Stopped!"
        r.publish('message', message)

        if recording == True:
            record_stop()

    return 'Signal Received'

@app.route('/status', methods=['GET'])
def get_status():
    global collecting
    global recording

    status = {}
    status['collecting'] = collecting
    status['recording'] = recording

    return json.dumps(status)

@app.route('/<path:path>', methods=['GET'])
def hello_world(path):
    return send_from_directory('video', path)


if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')

