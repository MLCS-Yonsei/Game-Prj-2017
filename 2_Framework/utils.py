# send_crest_request
import http.client
import csv 
import json
import os.path

import threading
import datetime
import redis

def send_crest_requset(url, flag, option):
    global standaloneWriter
    conn = http.client.HTTPConnection(url)
    conn.request("GET", "/crest/v1/api")

    res = conn.getresponse()

    data = json.loads(res.read().decode('utf8', "ignore").replace("'", '"'))

    if data["gameStates"]["mGameState"] > 1:
        if flag == 'standalone':
            file_path = './standalone.csv'

            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([str(datetime.now()),data])
        elif flag == 'crest-monitor':
            return data
        
    return data

def get_crest_data(target_ip, r):
    global crestThread
    
    dataLock = threading.Lock()
    with dataLock:
    # 데이터 가져오기
        crest_data = send_crest_requset(target_ip, "crest-monitor", {})
        
        gameState = crest_data['gameStates']['mGameState']

        if gameState > 1:
        # 게임 플레이중
            current_time = str(datetime.datetime.now())
            gamedata = {'current_time': current_time, 'gamedata': crest_data}

            r.hmset(target_ip, gamedata)

        else:
        # 플레이 종료
            print()
    
    # Set the next thread to happen
    POOL_TIME = 0.1

    crestThread = threading.Timer(POOL_TIME, get_crest_data, [target_ip, r])
    crestThread.start()
    
    return crestThread