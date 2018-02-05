import threading
import atexit
import redis
import datetime

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
yourThread = threading.Thread()

app = Flask(__name__, static_url_path='/video')

def create_app():
    def interrupt():
        global yourThread
        yourThread.cancel()

    def doStuff():
        global commonDataStruct
        global yourThread
        global recording

        with dataLock:
        # Do your stuff with commonDataStruct Here
            crest_data = send_crest_requset("192.168.0.2:9090", "crest-monitor", {})
            current_time = str(datetime.now())

            message = [current_time, crest_data]
            
            # print(current_time)
            r.publish('message', message)
        # Set the next thread to happen
        yourThread = threading.Timer(POOL_TIME, doStuff, ())
        yourThread.start()   

    def doStuffStart():
        # print("Start scanning ports..")
        # for i in range(2,255):
        #     url = "192.168.0." + str(i)
        #     print(scan_port(url,8080))


        # Do initialisation stuff here
        global yourThread
        # Create your thread
        yourThread = threading.Timer(POOL_TIME, doStuff, ())
        yourThread.start()

    # Initiate
    doStuffStart()
    # When you kill Flask (SIGTERM), clear the trigger for the next thread
    atexit.register(interrupt)
    return app

app = create_app()  

recording = False

camera_number = 2
resolution = "1280x720"

current_time = datetime.datetime.now()

vr = VideoRecorder(camera_number, resolution, current_time)

@app.route('/signal', methods=['GET'])
def get_signal():
    global recording
    
    if request.args.get('record') == "True" and recording == False:
        vr.start()
        recording = True
        message = "Recording Started!"
        r.publish('message', message)
    elif request.args.get('record') == "False" and recording == True:
        vr.stop()
        recording = False
        message = "Recording Stopped!"
        r.publish('message', message)

    return 'Signal Received'

@app.route('/<path:path>', methods=['GET'])
def hello_world(path):
    return send_from_directory('video', path)


if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')

