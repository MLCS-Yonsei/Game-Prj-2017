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

class overtakeChecker(mp.Process):

    def __init__(self,que,r,target_ip):
        # [공통] 기본설정
        super(overtakeChecker,self).__init__()
        self.queue = que
        self.r = r
        self.target_ip = target_ip

        # Variables
        self.r0_t0 = 0
        self.c = False
        self.status = False
        
    def get_rank(self, data):
        ranks = [info['mRacePosition'] for info in data["participants"]["mParticipantInfo"]]
        return ranks

    def run(self):
        #self.r.publish('results', self.target_ip + '/추우우우우월')
        while True:
            time.sleep(0.1)
            message = self.r.hgetall(self.target_ip)
            if message:
                data = {key.decode(): value.decode() for (key, value) in message.items()}
                gamedata = eval(data['gamedata'])
                
                # Codes
                ranks = self.get_rank(gamedata)
                r0_t1 = ranks[0]

                if self.r0_t0 != 0:
                    if self.r0_t0 > r0_t1:
                        # Overtaked
                        self.c = ranks.index(r0_t1 + 1)
                        self.status = True
                    elif self.r0_t0 < r0_t1:
                        # Overtaken
                        self.c = ranks.index(r0_t1 - 1)
                        self.status = False
                    else:
                        self.c = False

                if self.c:
                    c_name = gamedata["participants"]["mParticipantInfo"][self.c]["mName"]
                    
                    result = {}
                    result['target_ip'] = self.target_ip
                    result['flag'] = 'overtake'
                    result['data'] = {
                        'status': self.status,
                        'rank': r0_t1
                    }

                    self.r.hmset('results', result)

                self.r0_t0 = r0_t1

            