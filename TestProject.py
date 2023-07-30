import cv2
import time
import PoseEstimationMin as pm
import shot_detector
 
cap = cv2.VideoCapture('2.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) !=0:
        print(lmList[14])
        #right elbow
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        #right wrist
        cv2.circle(img, (lmList[16][1], lmList[16][2]), 15, (0, 255, 0), cv2.FILLED)

        #right shoulder
        cv2.circle(img, (lmList[12][1], lmList[12][2]), 15, (255, 0, 0), cv2.FILLED)
    
        detector.findAngle(img, 12, 14, 16, draw=True)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
 
    cv2.imshow("Image", img)
    cv2.waitKey(1)