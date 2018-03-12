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


while True:
    time.sleep(0.1)
    data = send_crest_requset(ip + ':' + str(args["port"]),'crest-monitor',{})
    lapdistance = data["participants"]["mParticipantInfo"][0]["mCurrentLapDistance"]
    if 790 < lapdistance < 810:
        print('300미터 앞 오른쪽 커브, 이어서 왼쪽 커브입니다')
    elif 3290 < lapdistance < 3310:
        print('이제부터 직선 구간입니다')
            
