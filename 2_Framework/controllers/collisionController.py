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

class collisionChecker(mp.Process):

    def __init__(self,que,r,target_ip):
        # [공통] 기본설정
        super(collisionChecker,self).__init__()
        self.queue = que
        self.r = r
        self.target_ip = target_ip

        self.channels = self.r.pubsub()
        self.channels.subscribe(self.target_ip)

        # Variables


    def run(self):
        while True:
            message = self.r.hgetall(self.target_ip)

            if message:
                data = {key.decode(): value.decode() for (key, value) in message.items()}
                gamedata = eval(data['gamedata'])
                
                # Codes
                current_time = str(datetime.datetime.now())
                
                result = {}
                result['current_time'] = current_time      
                result['target_ip'] = self.target_ip
                result['flag'] = 'collision'

                collision = gamedata["wheelsAndTyres"]["mTerrain"][0] != 0 and gamedata["wheelsAndTyres"]["mTerrain"][2] !=0
                velocity = sum( i*i for i in gamedata["motionAndDeviceRelated"]["mLocalVelocity"])

                if collision and velocity < 1.5 :
                    print('collision')
                    result['data'] = {
                    'collision' : True
                }
                

                self.r.hmset('results', result)

            time.sleep(0.1)
                
                