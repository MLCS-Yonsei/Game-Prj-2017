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

from overtakeController import overtakeChecker

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