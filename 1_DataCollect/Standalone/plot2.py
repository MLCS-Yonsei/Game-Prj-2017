# from rq import Queue
# from worker import conn

# from time import sleep
from utils import send_crest_requset, RepeatedTimer, scan_port

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--ip", type=str, default="192.168.0.49",
	help="IP address to get CREST data")

ap.add_argument("-p", "--port", type=int, default=8080,
	help="IP address to get CREST data")
args = vars(ap.parse_args())

ip = args["ip"]
datalist = ['mGameState','mSessionState','mRaceState','mCrashState','mEngineDamage','mAeroDamage','mAmbientTemperature','mTrackTemperature',
'mRainDensity','mWindSpeed','mWindDirectionX','mWindDirectionY','mCloudBrightness','mLapsInEvent','mTrackLocation','mTrackVariation','mTrackLength',
'mSpeed','mRpm','mBoostAmount','mBoostActive','mTyreFlags','mTerrain','mTyreY','mTyreRPS','mTyreSlipSpeed','mTyreTemp','mTyreGrip','mUnfilteredThrottle',
'mUnfilteredBrake','mUnfilteredSteering','mUnfilteredClutch','mBrack','mThrottle','mOrientation','mLocalVelocity','mWorldVelocity','mAngularVelocity',
'mLocalAcceleration','mWorldAcceleration','mExtentsCentre','mTyreHeightAboveGround','mTyreLateralStiffness','mTyreWear','mBrakeDamage',
'mSuspensionDamage','mViewedParticipantIndex','mNumParticipants','mParticipantInfo','mName','mIsActive','mWorldPosition','mCurrentSector','mCurrentLapDistance','mRacePosition',
'mLapsComleted','mCurrentLap']
# max_len = 0
# for i in range(len(datalist)):
#     if len(datalist[i])>= max_len:
#         max_len = len(datalist[i])

import curses
stdscr = curses.initscr()

lines=curses.LINES
cols=curses.COLS



def displayData(data):
    row_space = 10;i=0
    display = [data["gameStates"]["mGameState"], data["gameStates"]["mSessionState"], data["gameStates"]["mRaceState"], data["carDamage"]["mCrashState"],
    data["carDamage"]["mEngineDamage"], data["carDamage"]["mAeroDamage"], data["weather"]["mAmbientTemperature"], data["weather"]["mTrackTemperature"],
    data["weather"]["mRainDensity"], data["weather"]["mWindSpeed"], data["weather"]["mWindDirectionX"], data["weather"]["mWindDirectionY"],
    data["weather"]["mCloudBrightness"], data["eventInformation"]["mLapsInEvent"], data["eventInformation"]["mTrackLocation"], 
    data["eventInformation"]["mTrackVariation"], data["eventInformation"]["mTrackLength"], data["carState"]["mSpeed"], data["carState"]["mRpm"],
    data["carState"]["mBoostAmount"], data["carState"]["mBoostActive"], data["wheelsAndTyres"]["mTyreFlags"], data["wheelsAndTyres"]["mTerrain"],
    data["wheelsAndTyres"]["mTyreY"], data["wheelsAndTyres"]["mTyreRPS"], data["wheelsAndTyres"]["mTyreSlipSpeed"], data["wheelsAndTyres"]["mTyreTemp"],
    data["wheelsAndTyres"]["mTyreGrip"], data["unfilteredInput"]["mUnfilteredThrottle"], data["unfilteredInput"]["mUnfilteredBrake"],
    data["unfilteredInput"]["mUnfilteredSteering"], data["unfilteredInput"]["mUnfilteredClutch"], data["carState"]["mBrake"], data["carState"]["mThrottle"], 
    data["motionAndDeviceRelated"]["mOrientation"], data["motionAndDeviceRelated"]["mLocalVelocity"], data["motionAndDeviceRelated"]["mWorldVelocity"], 
    data["motionAndDeviceRelated"]["mAngularVelocity"], data["motionAndDeviceRelated"]["mLocalAcceleration"], data["motionAndDeviceRelated"]["mWorldAcceleration"],
    data["motionAndDeviceRelated"]["mExtentsCentre"], data["wheelsAndTyres"]["mTyreHeightAboveGround"], data["wheelsAndTyres"]["mTyreLateralStiffness"],
    data["wheelsAndTyres"]["mTyreWear"], data["wheelsAndTyres"]["mBrakeDamage"], data["wheelsAndTyres"]["mSuspensionDamage"], 
    data["participants"]["mViewedParticipantIndex"], data["participants"]["mNumParticipants"],"mParticipantInfo", data["participants"]["mParticipantInfo"][i]["mName"],
    data["participants"]["mParticipantInfo"][i]["mIsActive"], data["participants"]["mParticipantInfo"][i]["mWorldPosition"],
    data["participants"]["mParticipantInfo"][i]["mCurrentSector"], data["participants"]["mParticipantInfo"][i]["mCurrentLapDistance"],
    data["participants"]["mParticipantInfo"][i]["mRacePosition"], data["participants"]["mParticipantInfo"][i]["mLapsCompleted"],
    data["participants"]["mParticipantInfo"][i]["mCurrentLap"]]
    return display  

curses.cbreak()
stdscr.keypad(1)
#stdscr.addstr(0, 0, str(max_len))
stdscr.addstr(0,20,"Hit 'q' to quit")
stdscr.refresh()

for j in range (len(datalist)):
    stdscr.addstr((1+(j%(lines-1))),(j//(lines-1))*(cols//(len(datalist)//(lines-1)+1)),datalist[j])

key = ''
while key != ord('q'):
    #key = stdscr.getch()


    data = send_crest_requset(ip + ':' + str(args["port"]),'crest-monitor',{})

    display = displayData(data)

    for j in range (len(datalist)):
        stdscr.addstr((1+(j%(lines-1))),len(datalist[j])+1+(j//(lines-1))*(cols//(len(datalist)//(lines-1)+1)),str(display[j]))

    stdscr.refresh()


curses.endwin()



# def key(event):
#     stop = True
#     global rt
#     if event.char == 's':
#         if scan_port(ip,args["port"]) == True and stop == True:
#             rt = RepeatedTimer(0.1, main) # it auto-starts, no need of rt.start()
#             label1.config(text='Getting Data From CREST, Press q to quit.')
#             stop = False
#     elif event.char == 'q':
#         rt.stop()
#         label1.config(text='Stop Getting Data From CREST, Press s to start.')
#         stop = True