#installing relevant packages

!pip install opencv-python
!pip install mediapipe


import cv2
import mediapipe as mp
import csv
import time
from datetime import datetime
import re
import pandas as pd
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

video_path = 'file:///Users/ashmohan/Downloads/Ashwin_Video.mp4'
cap = cv2.VideoCapture(video_path)

landmarks_data = []
DataOutput = pd.DataFrame()


#Get the FPS of the video.
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0

LabelName = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
             'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
             'MIDDLE_FINGER_TIP','RING_FINGER_MCP','RING_FINGER_PIP','RING_FINGER_DIP','RING_LFINGER_TIP','PINKY_MCP',
             'PINKY_PIP','PINKY_DIP','PINKY_TIP']
LabelName2= LabelName + LabelName
ID = list(range(0,21))
ID2 = list(range(0,21)) +list(range(0,21))

pose_keypoints = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
]

hand_keypoints = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

with mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=True) as pose:
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,model_complexity=1, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('MediaPipe Pose and Hands', frame_bgr)


            if cv2.waitKey(1) & 0xFF == 27:
                break

            timestamp = frame_count / fps
            frame_count += 1

            # Add pose keypoint data
            if pose_results.pose_world_landmarks:
                pose_data = []
                for lm in pose_results.pose_world_landmarks.landmark:
                    pose_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmarks_data.append(['pose', timestamp] + pose_data)
                t=1


            # Add hand keypoint data
            ## data classification
            ## whether the hand is exsit in image
            if hand_results.multi_hand_world_landmarks:

                Time = datetime.now().strftime("%H:%M:%S.%f")
                h, w, c = frame.shape
                dataSlice = str(hand_results.multi_hand_world_landmarks)

                xCor = re.findall(r'x: .........', dataSlice)
                xdataArray = []
                for id, value in enumerate(xCor):
                    Datastr = xCor[id]
                    Datastr = Datastr[3:]
                    try:
                        xdataArray.append(round(float(Datastr), 5))
                    except:
                        xdataArray.append(0.0001)


                yCor = re.findall(r'y: .........', dataSlice)
                ydataArray = []
                for id, value in enumerate(yCor):
                    Datastr = yCor[id]
                    Datastr = Datastr[3:]
                    try:
                        ydataArray.append(round(float(Datastr), 5))
                    except:
                        ydataArray.append(0.0001)

                zCor = re.findall(r'z: .........', dataSlice)
                zdataArray = []
                for id, value in enumerate(zCor):
                    Datastr = zCor[id]
                    Datastr = Datastr[3:]
                    try:
                        zdataArray.append(round(float(Datastr), 5))
                    except:
                        zdataArray.append(0.0001)

                # tt = print(results.multi_handedness)  ###estimated probability of the predicted handedness------not accurate at all
                num_hands = len(hand_results.multi_handedness)
                if num_hands == 1:  ### if only one hand show
                    # handedness0 = str(results.multi_handedness[0].classification)
                    # handedness0 = str(re.findall(r'label: ..', handedness0))
                    # handedness0 = list(handedness0[-3])
                    handedness0 = str(hand_results.multi_handedness[0].classification)
                    handedness0 = str(re.findall(r'index: .', handedness0))
                    handedness0 = re.findall(r"\d", handedness0)
                    handedness0 = float(handedness0[0])

                    if handedness0 == 0:
                        hand0 = "Left"  ###estimated probability of the predicted handedness------not accurate at all
                    else:
                        hand0 = "Right"  ###estimated probability of the predicted handedness------not accurate at all
                    HandStr = [hand0] * 21

                    d = {'XCor': xdataArray, 'YCor': ydataArray, 'ZCor': zdataArray, 'ID': list(range(0, 21)),
                         'LabelName': LabelName, 'Handedness': HandStr, 'TimeStamp': np.full(21, Time)}
                    try:
                        DataMatrix = pd.DataFrame(data=d)
                    except:
                        t3 = 1




                elif num_hands == 2:  ### if two hands show
                    handedness0 = str(hand_results.multi_handedness[0].classification)
                    handedness0 = str(re.findall(r'index: .', handedness0))
                    handedness0 = re.findall(r"\d", handedness0)
                    handedness0 = float(handedness0[0])
                    handedness1 = str(hand_results.multi_handedness[1].classification)
                    handedness1 = str(re.findall(r'index: .', handedness1))
                    handedness1 = re.findall(r"\d", handedness1)
                    handedness1 = float(handedness1[0])

                    if int(handedness0) != int(handedness1):  ### if both left (0) and right(1) hand showed
                        if int(handedness0) == 0:
                            hand0 = "Left"  ###estimated probability of the predicted handedness------not accurate at all
                            hand1 = "Right"
                            HandStr = [hand0] * 21 + [hand1] * 21
                        else:
                            hand0 = "Right"  ###estimated probability of the predicted handedness------not accurate at all
                            hand1 = "Left"
                            HandStr = [hand0] * 21 + [hand1] * 21

                    else:
                        HandStr = [hand0] * 21 + [hand1] * 21
                        continue  #### do nothing

                    d = {'XCor(m)': xdataArray, 'YCor(m)': ydataArray, 'ZCor(m)': zdataArray, 'ID': ID2,
                         'LabelName': LabelName2, 'Handedness': HandStr, 'TimeStamp': np.full(42, Time)}
                    T=HandStr
                    T2= np.full(42, Time)
                    try:
                        DataMatrix = pd.DataFrame(data=d)
                    except:
                        t3=1

                else:
                    continue  # do nothing

                DataOutput = pd.concat([DataOutput, DataMatrix], ignore_index=True)

                currentDateAndTime = datetime.now()
                currentTimeHMS = currentDateAndTime.strftime("%H_%M")

                # print("Time stamp is", currentTimeHMS)
                DataOutput.to_csv('HandDataAt' + currentTimeHMS + '.csv')
        cap.release()
        cv2.destroyAllWindows()

t=1
#Write keypoint data to CSV file
header = ['type', 'timestamp'] + [f'{prefix}{name}{coord}' for prefix, names in zip(['pose', 'hand'], [pose_keypoints, hand_keypoints]) for i, name in enumerate(names, start=1) for coord in ['x', 'y', 'z', 'visibility'] if not (prefix == 'hand' and coord == 'visibility')]

csv_filename = 'landmarks_data.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerows(landmarks_data)# Paste your MediaPipe landmark tracking code here
