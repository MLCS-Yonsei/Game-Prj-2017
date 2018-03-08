# send_crest_request
import http.client
import csv 
import json
import os.path

# RepeatedTimer
from threading import Timer

# Port Scanner
#!/usr/bin/env python
import socket
import subprocess
import sys
from datetime import datetime


def send_crest_requset(url, flag, option):
    global standaloneWriter
    conn = http.client.HTTPConnection(url)
    conn.request("GET", "/crest/v1/api")

    res = conn.getresponse()

    data = json.loads(res.read().decode('utf8', "ignore").replace("'", '"'))

    if data["gameStates"]["mGameState"] == 3:
        if flag == 'standalone':
            file_path = './standalone.csv'

            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([str(datetime.now()),data])
        elif flag == 'crest-monitor':
            return data
        elif flag == 'map-labeller':
            pressed_key = option["pressed_key"]
            track_name = data["eventInformation"]["mTrackLocation"]
            track_length = data["eventInformation"]["mTrackLength"]

            current_player = data["participants"]["mParticipantInfo"][0]
            current_world_position = current_player["mWorldPosition"]

            current_world_position_x = current_world_position[0]
            current_world_position_y = current_world_position[1]
            current_world_position_z = current_world_position[2]

            current_lap_distance = current_player["mCurrentLapDistance"]

            file_path = './track_data/' + track_name + '.csv'

            if not os.path.isfile(file_path):
                with open(file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Track Name","Track Length","Label","Current Lap Distance","Current World X","Current World Y","Current World Z"])

            if pressed_key != "":
                with open(file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([track_name,track_length,pressed_key,current_lap_distance,current_world_position_x,current_world_position_y,current_world_position_z])

    return data

def scan_port(ip,port):
    # Clear the screen
    # subprocess.call('clear', shell=True)

    # Ask for input
    remoteServer    = ip
    remoteServerIP  = socket.gethostbyname(remoteServer)

    # Print a nice banner with information on which host we are about to scan
    print("-" * 60)
    print("Please wait, scanning remote host", remoteServerIP)
    print("-" * 60)

    # Check what time the scan started
    t1 = datetime.now()

    # Using the range function to specify ports (here it will scans all ports between 1 and 1024)

    # We also put in some error handling for catching errors

    try:
        for port in range(port,port+1):  
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((remoteServerIP, port))
            if result == 0:
                print("Port {}: 	 Open".format(port))
            sock.close()

    except KeyboardInterrupt:
        print("You pressed Ctrl+C")
        sys.exit()

    except socket.gaierror:
        print('Hostname could not be resolved. Exiting')
        sys.exit()

    except socket.error:
        print("Couldn't connect to server")
        sys.exit()

    # Checking the time again
    t2 = datetime.now()

    # Calculates the difference of time, to see how long it took to run the script
    total =  t2 - t1

    # Printing the information to screen
    print('Scanning Completed in: ', total)

    if result == 0:
        return True
    else:
        return False

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


