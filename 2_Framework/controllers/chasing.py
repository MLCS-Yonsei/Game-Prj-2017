from utils import send_crest_requset, RepeatedTimer, scan_port
import argparse

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
    data = send_crest_requset(ip + ':' + str(args["port"]),'crest-monitor',{})
    distance, lap1, lap2 = get_distance(data)
    if len(recent_position) == 20:
        recent_position = recent_position[1:]
        recent_position.append(distance)
        if lap1 == lap2 and distance < 60 and recent_position[19]-recent_position[0] < -20:
            pass

    elif len(recent_position) < 20:
        recent_position.append(distance)
