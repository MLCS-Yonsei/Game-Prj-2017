from utils import send_crest_requset, RepeatedTimer, scan_port
import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--ip", type=str, default="mlcs.yonsei.ac.kr",
	help="IP address to get CREST data")

ap.add_argument("-p", "--port", type=int, default=8080,
	help="IP address to get CREST data")
args = vars(ap.parse_args())

ip = args["ip"]

def get_distance(data):
    ranks = [info['mRacePosition'] for info in data["participants"]["mParticipantInfo"]]
    distance = data["participants"]["mParticipantInfo"][ranks.index(ranks[0]-1)]["mCurrentLapDistance"] - data["participants"]["mParticipantInfo"][ranks.index(ranks[0])]["mCurrentLapDistance"]
    lap1 = data["participants"]["mParticipantInfo"][ranks.index(ranks[0]-1)]["mCurrentLap"]
    lap2 = data["participants"]["mParticipantInfo"][ranks.index(ranks[0])]["mCurrentLap"]

    return distance, lap1, lap2


recent_position = []
while True:
    time.sleep(0.1)
    data = send_crest_requset(ip + ':' + str(args["port"]),'crest-monitor',{})
    distance, lap1, lap2 = get_distance(data)
    if len(recent_position) == 20:
        recent_position = recent_position[1:]
        recent_position.append(distance)
        if lap1 == lap2 and distance < 60 and recent_position[19]-recent_position[0] < -20:
            pass            

    elif len(recent_position) < 20:
        recent_position.append(distance)
'''
import subprocess 
import multiprocessing as mp
from threading import Thread
from multiprocessing import Pool
from queue import Empty

import time
import datetime
import os
import signal

import redis

class chasing(mp.Process):

    def __init__(self,que,r,target_ip):
        # [공통] 기본설정
        super(chasing,self).__init__()
        self.queue = que
        self.r = r
        self.target_ip = target_ip

        # Variables
        self.recent_position = []
        self.c = False
        self.status = False
        
    def get_distance(self,data):
        ranks = [info['mRacePosition'] for info in data["participants"]["mParticipantInfo"]]
        distance = data["participants"]["mParticipantInfo"][ranks.index(ranks[0]-1)]["mCurrentLapDistance"] - data["participants"]["mParticipantInfo"][ranks.index(ranks[0])]["mCurrentLapDistance"]
        lap1 = data["participants"]["mParticipantInfo"][ranks.index(ranks[0]-1)]["mCurrentLap"]
        lap2 = data["participants"]["mParticipantInfo"][ranks.index(ranks[0])]["mCurrentLap"]

        return distance, lap1, lap2

    def run(self):

        while True:
            time.sleep(0.1)
            message = self.r.hgetall(self.target_ip)

            if message:
                data = {key.decode(): value.decode() for (key, value) in message.items()}
                gamedata = eval(data['gamedata'])
                
                # Codes
                distance, lap1, lap2 = self.get_distance(gamedata)

                if len(self.recent_position) == 20:
                    self.recent_position = self.recent_position[1:]
                    self.recent_position.append(distance)
                    if lap1 == lap2 and distance < 60 and recent_position[19]-recent_position[0] < -20:
                        ranks = [info['mRacePosition'] for info in data["participants"]["mParticipantInfo"]]
                        self.c = ranks.index(ranks[0])
                        self.status = True
                    else:
                        self.c = False
                        self.status = False
                
                elif len(self.recent_position) < 20:
                    recent_position.append(distance)

                if self.c:
                    c_name = gamedata["participants"]["mParticipantInfo"][self.c]["mName"]
                    
                    result = {}
                    result['target_ip'] = self.target_ip
                    result['flag'] = 'chase'
                    result['data'] = {
                        'status': self.status
                    }

                    self.r.hmset('results', result)

'''
            