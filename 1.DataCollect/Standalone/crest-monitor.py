from rq import Queue
from worker import conn

from time import sleep
from utils import send_crest_requset, RepeatedTimer, scan_port

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--ip", type=str, default="192.168.0.2",
	help="IP address to get CREST data")

ap.add_argument("-p", "--port", type=int, default=8080,
	help="IP address to get CREST data")
args = vars(ap.parse_args())

ip = args["ip"]

import curses
stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)

stdscr.addstr(0,10,"Hit 'q' to quit")
stdscr.refresh()

stdscr.addstr(1, 0, "gameStates => ")
stdscr.addstr(1, 15, "mGameState")
stdscr.addstr(2, 15, "mSessionState")
stdscr.addstr(3, 15, "mRaceState")

stdscr.addstr(17, 0, "participants => ")
stdscr.addstr(17, 15, "mViewedParticipantIndex")
stdscr.addstr(18, 15, "mNumParticipants")
stdscr.addstr(19, 15, "mParticipantInfo")
stdscr.addstr(19, 0, "↓↓↓↓↓↓")

stdscr.addstr(1, 45, "eventInformation => ↓")
stdscr.addstr(2, 45, "mLapsInEvent")
stdscr.addstr(3, 45, "mTrackLocation")
stdscr.addstr(4, 45, "mTrackVariation")
stdscr.addstr(5, 45, "mTrackLength")

stdscr.addstr(4, 45, "mTrackVariation")
stdscr.addstr(5, 45, "mTrackLength")

stdscr.addstr(1, 85, "unfilteredInput  => ↓")
stdscr.addstr(2, 85, "mUnfilteredThrottle")
stdscr.addstr(3, 85, "mUnfilteredBrake")
stdscr.addstr(4, 85, "mUnfilteredSteering")
stdscr.addstr(5, 85, "mUnfilteredClutch")

stdscr.addstr(7, 45, "carState => ↓")
stdscr.addstr(8, 45, "mSpeed / mRpm")
stdscr.addstr(9, 45, "mBoostAmount")
stdscr.addstr(8, 85, "mBrake")
stdscr.addstr(9, 85, "mThrottle")

stdscr.addstr(1, 115, "motionAndDeviceRelated => ↓")
stdscr.addstr(2, 115, "mOrientation")
stdscr.addstr(3, 115, "mLocalVelocity")
stdscr.addstr(4, 115, "mWorldVelocity")
stdscr.addstr(5, 115, "mAngularVelocity")
stdscr.addstr(6, 115, "mLocalAcceleration")
stdscr.addstr(7, 115, "mWorldAcceleration")
stdscr.addstr(8, 115, "mExtentsCentre")

stdscr.addstr(5, 0, "carDamage => ")
stdscr.addstr(5, 15, "mCrashState")
stdscr.addstr(7, 15, "mAeroDamage")
stdscr.addstr(6, 15, "mEngineDamage")

stdscr.addstr(9, 0, "weather => ")
stdscr.addstr(9, 15, "mAmbientTemperature")
stdscr.addstr(10, 15, "mTrackTemperature")
stdscr.addstr(11, 15, "mRainDensity")
stdscr.addstr(12, 15, "mWindSpeed")
stdscr.addstr(13, 15, "mWindDirectionX")
stdscr.addstr(14, 15, "mWindDirectionY")
stdscr.addstr(15, 15, "mCloudBrightness")

stdscr.addstr(12, 45, "wheelsAndTyres => ↓")
stdscr.addstr(13, 45, "mTyreFlags")
stdscr.addstr(14, 45, "mTerrain")
stdscr.addstr(15, 45, "mTyreY")
stdscr.addstr(16, 45, "mTyreRPS")
stdscr.addstr(17, 45, "mTyreSlipSpeed")
stdscr.addstr(18, 45, "mTyreTemp")
stdscr.addstr(19, 45, "mTyreGrip")

stdscr.addstr(13, 120, "mTyreHeightAboveGround")
stdscr.addstr(14, 120, "mTyreLateralStiffness")
stdscr.addstr(15, 120, "mTyreWear")
stdscr.addstr(16, 120, "mBrakeDamage")
stdscr.addstr(17, 120, "mSuspensionDamage")
stdscr.addstr(19, 120, "Brake, TyreLayer, TyreCarcass, TyreRim, TyirInternalAir Temp 생략")


def displayData(data):
    row_space = 10
    if True:
        display = data["gameStates"]["mGameState"]
        stdscr.addstr(1, 40, str(display))

        display = data["gameStates"]["mSessionState"]
        stdscr.addstr(2, 40, str(display))

        display = data["gameStates"]["mRaceState"]
        stdscr.addstr(3, 40, str(display))

        display = data["participants"]["mViewedParticipantIndex"]
        stdscr.addstr(17, 40, str(display))

        display = data["participants"]["mNumParticipants"]
        stdscr.addstr(18, 40, str(display))

        for i in range(0,data["participants"]["mNumParticipants"]):
            if i<=9:
                display = "Particiapnt #" + str(i+1)
                stdscr.addstr(row_space + 11 + 6 * int(i / 3), 0 + 65 * (i % 3), str(display))

                display = "Name : " + data["participants"]["mParticipantInfo"][i]["mName"]
                stdscr.addstr(row_space + 12 + 6 * int(i / 3), 20 + 65 * (i % 3), str(display))

                display = "mIsActive"
                stdscr.addstr(row_space + 12 + 6 * int(i / 3), 0 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mIsActive"]
                stdscr.addstr(row_space + 13 + 6 * int(i / 3), 0 + 65 * (i % 3), str(display))

                display = "mWorldPosition"
                stdscr.addstr(row_space + 12 + 6 * int(i / 3), 15 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mWorldPosition"] 
                stdscr.addstr(row_space + 13 + 6 * int(i / 3), 15 + 65 * (i % 3), str(display))

                display = "mCurrentSector"
                stdscr.addstr(row_space + 12 + 6 * int(i / 3), 45 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mCurrentSector"] 
                stdscr.addstr(row_space + 13 + 6 * int(i / 3), 45 + 65 * (i % 3), str(display))

                display = "mCurrentLapDistance"
                stdscr.addstr(row_space + 14 + 6 * int(i / 3), 0 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mCurrentLapDistance"]
                stdscr.addstr(row_space + 15 + 6 * int(i / 3), 0 + 65 * (i % 3), str(display))

                display = "mRacePosition"
                stdscr.addstr(row_space + 14 + 6 * int(i / 3), 15 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mRacePosition"] 
                stdscr.addstr(row_space + 15 + 6 * int(i / 3), 15 + 65 * (i % 3), str(display))

                display = "mLapsCompleted"
                stdscr.addstr(row_space + 14 + 6 * int(i / 3), 30 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mLapsCompleted"]
                stdscr.addstr(row_space + 15 + 6 * int(i / 3), 30 + 65 * (i % 3), str(display))

                display = "mCurrentLap"
                stdscr.addstr(row_space + 14 + 6 * int(i / 3), 45 + 65 * (i % 3), str(display))

                display = data["participants"]["mParticipantInfo"][i]["mCurrentLap"] 
                stdscr.addstr(row_space + 15 + 6 * int(i / 3), 45 + 65 * (i % 3), str(display))

        display = data["eventInformation"]["mLapsInEvent"]
        stdscr.addstr(2, 65, str(display))

        display = data["eventInformation"]["mTrackLocation"]
        stdscr.addstr(3, 65, str(display))

        display = data["eventInformation"]["mTrackVariation"]
        stdscr.addstr(4, 65, str(display))

        display = data["eventInformation"]["mTrackLength"]
        stdscr.addstr(5, 65, str(display))

        display = data["unfilteredInput"]["mUnfilteredThrottle"]
        stdscr.addstr(2, 105, str(display))

        display = data["unfilteredInput"]["mUnfilteredBrake"]
        stdscr.addstr(3, 105, str(display))

        display = data["unfilteredInput"]["mUnfilteredSteering"]
        stdscr.addstr(4, 105, str(display))

        display = data["unfilteredInput"]["mUnfilteredClutch"]
        stdscr.addstr(5, 105, str(display))

        display = data["carState"]["mSpeed"]
        stdscr.addstr(8, 65, str(display))

        display = data["carState"]["mRpm"]
        stdscr.addstr(8, 75, str(display))

        display = data["carState"]["mBoostAmount"]
        stdscr.addstr(9, 65, str(display))

        display = data["carState"]["mBoostActive"]
        stdscr.addstr(9, 70, str(display))

        display = data["carState"]["mBrake"]
        stdscr.addstr(8, 105, str(display))

        display = data["carState"]["mThrottle"]
        stdscr.addstr(9, 105, str(display))
    
        display = data["motionAndDeviceRelated"]["mOrientation"]
        stdscr.addstr(2, 135, str(display))

        display = data["motionAndDeviceRelated"]["mLocalVelocity"]
        stdscr.addstr(3, 135, str(display))

        display = data["motionAndDeviceRelated"]["mWorldVelocity"]
        stdscr.addstr(4, 135, str(display))

        display = data["motionAndDeviceRelated"]["mAngularVelocity"]
        stdscr.addstr(5, 135, str(display))

        display = data["motionAndDeviceRelated"]["mLocalAcceleration"]
        stdscr.addstr(6, 135, str(display))

        display = data["motionAndDeviceRelated"]["mWorldAcceleration"]
        stdscr.addstr(7, 135, str(display))

        display = data["motionAndDeviceRelated"]["mExtentsCentre"]
        stdscr.addstr(8, 135, str(display))

        display = data["carDamage"]["mCrashState"]
        stdscr.addstr(5, 35, str(display))

        display = data["carDamage"]["mAeroDamage"]
        stdscr.addstr(7, 35, str(display))

        display = data["carDamage"]["mEngineDamage"]
        stdscr.addstr(6, 35, str(display))

        display = data["weather"]["mAmbientTemperature"]
        stdscr.addstr(9, 35, str(display))

        display = data["weather"]["mTrackTemperature"]
        stdscr.addstr(10, 35, str(display))

        display = data["weather"]["mRainDensity"]
        stdscr.addstr(11, 35, str(display))

        display = data["weather"]["mWindSpeed"]
        stdscr.addstr(12, 35, str(display))

        display = data["weather"]["mWindDirectionX"]
        stdscr.addstr(13, 35, str(display))

        display = data["weather"]["mWindDirectionY"]
        stdscr.addstr(14, 35, str(display))

        display = data["weather"]["mCloudBrightness"]
        stdscr.addstr(15, 35, str(display))


    display = data["wheelsAndTyres"]["mTyreFlags"]
    stdscr.addstr(13, 60, str(display))

    display = data["wheelsAndTyres"]["mTerrain"]
    stdscr.addstr(14, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreY"]
    stdscr.addstr(15, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreRPS"]
    stdscr.addstr(16, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreSlipSpeed"]
    stdscr.addstr(17, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreTemp"]
    stdscr.addstr(18, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreGrip"]
    stdscr.addstr(19, 60, str(display))

    display = data["wheelsAndTyres"]["mTyreHeightAboveGround"]
    stdscr.addstr(13, 140, str(display))

    display = data["wheelsAndTyres"]["mTyreLateralStiffness"]
    stdscr.addstr(14, 140, str(display))

    display = data["wheelsAndTyres"]["mTyreWear"]
    stdscr.addstr(15, 140, str(display))

    display = data["wheelsAndTyres"]["mBrakeDamage"]
    stdscr.addstr(16, 140, str(display))

    display = data["wheelsAndTyres"]["mSuspensionDamage"]
    stdscr.addstr(17, 140, str(display))


    # display = data["motionAndDeviceRelated"]["mExtentsCentre"]
    # stdscr.addstr(8, 135, str(display))
    
    

key = ''
while key != ord('q'):
    
    data = send_crest_requset(ip + ':' + str(args["port"]),'crest-monitor',{})

    displayData(data)
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
    
