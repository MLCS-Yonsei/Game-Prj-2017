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

class lapDistanceChecker(mp.Process):

    def __init__(self,que,r,target_ip):
        # [공통] 기본설정
        super(lapDistanceChecker,self).__init__()
        self.queue = que
        self.r = r
        self.target_ip = target_ip

        self.channels = self.r.pubsub()
        self.channels.subscribe(self.target_ip)

        # Variables


    def run(self):
        t = 0
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
                result['flag'] = 'lapDistance'

                lap_length = gamedata["eventInformation"]["mTrackLength"] # 랩 길이
                lap_completed = gamedata["participants"]["mParticipantInfo"][0]["mLapsCompleted"]
                lap_distance = gamedata["participants"]["mParticipantInfo"][0]["mCurrentLapDistance"] + lap_length * lap_completed
                racestate= gamedata["gameStates"]["mRaceState"]

                result['data'] = {
                    'lapDistance' : lap_distance,
                    'event' : ''
                }
                
                if t == 0 and racestate == 2 :
                    print('start')
                    t +=1
                    result['data']['event'] = 'start'

                elif 90 < lap_distance < 110 :
                    print('터널입니다')
                    result['data']['event'] = 'tunnel'

                elif 790 < lap_distance < 810 :
                    print('앞에 급한 커브입니다')
                    result['data']['event'] = 'deep_curve'
                
                elif 1240 < lap_distance < 1260 :
                    print('앞에 급한 커브입니다')
                    result['data']['event'] = 'deep_curve'

                elif 1890 < lap_distance < 1910 :
                    print('앞에 완만한 s자 커브입니다')
                    result['data']['event'] = 'curve'

                elif 2490 < lap_distance < 2510 :
                    print('앞에 완만한 s자 커브입니다')
                    result['data']['event'] = 'curve'

                elif 3290 < lap_distance < 3310 :
                    print('이제부터 직선 구간입니다')
                    result['data']['event'] = 'straight'

                elif 4800 < lap_distance < 4810 :
                    print('거의 다 왔습니다')
                    result['data']['event'] = 'finish'

                # if racestate == 3 and t ==1 :
                #     t += 1
                #     print('끝났습니다')
                #     result['data']['event'] = 'finish'

                if result['data']['event'] != '':
                    self.r.hmset('results', result)

            time.sleep(0.1)
                
                