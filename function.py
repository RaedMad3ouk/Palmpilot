import pickle
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pyautogui
import time
from collections import deque

model = tf.keras.models.load_model('model/model.keras')
gesture_labels = pickle.load(open('model/gesture_labels.pickle','rb'))

SEQ_LENGTH = 60
COOLDOWN_SWITCH = 2.0
MOUSE_SENSITIVITY = 0.6
FRAME_MARGIN = 50
CLICK_THRESHOLD = 40

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

mode = "function"
last_switch_time = 0
sequence = deque(maxlen=SEQ_LENGTH)
prev_gesture = None
last_action_time = 0
screen_w, screen_h = pyautogui.size()
prev_index_touch = False
prev_middle_touch = False
drag_active = False
click_timestamps = []
drag_start_time = 0.0

def dist(lm1,lm2,fw,fh):
    x1,y1 = int(lm1.x*fw),int(lm1.y*fh)
    x2,y2 = int(lm2.x*fw),int(lm2.y*fh)
    return np.hypot(x2-x1,y2-y1)

def connected(lm1,lm2,fw,fh,th=CLICK_THRESHOLD):
    return dist(lm1,lm2,fw,fh)<th

while True:
    ret,frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame,1)
    fh,fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    now = time.time()

    if mode=="function":
        if res.multi_hand_landmarks:
            lm_list = res.multi_hand_landmarks[0].landmark
            feat = []
            for lm in lm_list:
                feat += [lm.x,lm.y,lm.z]
            sequence.append(feat)
            cv2.drawLandmarks = getattr(cv2, 'drawLandmarks', None)  # ensure attribute
            mp.solutions.drawing_utils.draw_landmarks(frame,res.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS)
            if len(sequence)==SEQ_LENGTH:
                inp = np.expand_dims(sequence,axis=0)
                preds = model.predict(inp,verbose=0)[0]
                idx = np.argmax(preds)
                conf = preds[idx]
                gesture = gesture_labels[idx]
                cv2.putText(frame,f"{gesture} ({conf:.2%})",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                if gesture=="switch" and now-last_switch_time>COOLDOWN_SWITCH:
                    mode="mouse"
                    last_switch_time=now
                    sequence.clear()
                    time.sleep(0.5)
                    continue
                if gesture!=prev_gesture or now-last_action_time>1.0:
                    if gesture=="scroll up":   pyautogui.scroll(300)
                    if gesture=="scroll down": pyautogui.scroll(-300)
                    if gesture=="zoom in":     pyautogui.hotkey('ctrl','+')
                    if gesture=="zoom out":    pyautogui.hotkey('ctrl','-')
                    prev_gesture=gesture
                    last_action_time=now
        cv2.putText(frame,"MODE: FUNCTION",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),3)
    else:
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            ix,iy = int(lm[8].x*fw),int(lm[8].y*fh)
            sx = np.interp(ix,[FRAME_MARGIN,fw-FRAME_MARGIN],[0,screen_w])*MOUSE_SENSITIVITY
            sy = np.interp(iy,[FRAME_MARGIN,fh-FRAME_MARGIN],[0,screen_h])*MOUSE_SENSITIVITY
            sx,sy = np.clip(sx,0,screen_w),np.clip(sy,0,screen_h)
            pyautogui.moveTo(sx,sy)
            cv2.rectangle(frame,(FRAME_MARGIN,FRAME_MARGIN),(fw-FRAME_MARGIN,fh-FRAME_MARGIN),(100,255,100),2)
            cv2.circle(frame,(ix,iy),8,(255,0,255),-1)
            index_touch = connected(lm[8],lm[4],fw,fh)
            middle_touch = connected(lm[12],lm[4],fw,fh)
            if index_touch and not prev_index_touch:
                pyautogui.click(button='right')
            prev_index_touch = index_touch
            if middle_touch and not prev_middle_touch:
                click_timestamps.append(now)
                click_timestamps=[t for t in click_timestamps if now-t<0.5]
                if len(click_timestamps)>=2:
                    pyautogui.doubleClick()
                    click_timestamps.clear()
                else:
                    pyautogui.click(button='left')
                drag_start_time = now
            prev_middle_touch = middle_touch
            if middle_touch and now-drag_start_time>0.7 and not drag_active:
                pyautogui.mouseDown()
                drag_active=True
            if not middle_touch and drag_active:
                pyautogui.mouseUp()
                drag_active=False
            if lm[4].y<lm[3].y<lm[2].y and all(lm[4].y<lm[i].y for i in (8,12,16,20)):
                if now-last_switch_time>COOLDOWN_SWITCH:
                    mode="function"
                    last_switch_time=now
                    sequence.clear()
                    time.sleep(0.5)
                    continue
        cv2.putText(frame,"MODE: MOUSE",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),3)

    cv2.imshow("Unified Gesture Control",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
