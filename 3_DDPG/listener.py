import time
import json

from socket import *
csock = socket(AF_INET, SOCK_DGRAM)
csock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
csock.bind(('', 6001))

status = False

i = 0
while True:
    if status == False:
        csock.sendto('Connect'.encode(), ('192.168.0.49',54545)) # 대상 서버 , 목적지 포트
        s, addr = csock.recvfrom(1024)
        print('Connected!')
        status = True
    else:
        print(i)
        i = i + 1

        n = i / 20
        if n < 0.5:
            controlState = {
                'acc': False,
                'brake': False,
                'steer': n
            }
        else:
            controlState = {
                'acc': True,
                'brake': False,
                'steer': n
            }
        
        json_str = json.dumps(controlState)

        csock.sendto(json_str.encode(), ('192.168.0.49',54545)) # 대상 서버 , 목적지 포트
