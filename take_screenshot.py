import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            #accessing the landmarks by their position
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            #array to hold true or false if finger is folded    
            finger_fold_status =[]
            for tip in finger_tips:
                #getting the landmark tip position and drawing purple circle
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                cv2.circle(img, (x,y), 15, (204, 51, 255), cv2.FILLED)

                #if finger folded changing color to pink
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (255, 51,204 ), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

             #checking if all fingers are folded
            if all(finger_fold_status):
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite("in_memory_to_disk.png", image)

                #take a screenshot directly to disk
                pyautogui.screenshot("straight_to_disk.png")

                
                image = cv2.imread("straight_to_disk.png")
                cv2.imshow("Screenshot", imutils.resize(image, width=600))


            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((255, 51, 0),2,2),
            mp_draw.DrawingSpec((102, 51, 0),4,2))
    

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()