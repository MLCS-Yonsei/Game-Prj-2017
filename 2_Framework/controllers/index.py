import sys
sys.path.insert(0, '../routes')

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
        super(overtakeChecker,self).__init__()
        self.queue = que
        self.r = r
        self.target_ip = target_ip

        self.channels = self.r.pubsub()
        self.channels.subscribe([self.target_ip])

    def run(self):
        #print(self.target_ip, '추우우우우월!')
        while True:
            message = self.channels.get_message()
            if message:
                self.r.publish('results', self.target_ip + '/추우우우우월')
        

class controller():

    def __init__(self):
        self.jobs = []
        self.queues = []

    def checkOvertake(self, r, target_ip):
        q = mp.Queue()
        job = overtakeChecker(q,r,target_ip)
        self.queues.append(q)
        self.jobs.append(job)
        job.start()