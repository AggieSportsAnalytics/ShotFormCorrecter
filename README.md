---
title: Basketball Freethrow Shotfrom Corrector
description: Determines if a basketball player releases at the correct angle and follows through with the help of computer vision
repository: AggieSportsAnalytics/ShotFormCorrecter

date: 2023-09-29
published: true
---

### 🏁 Determines if a basketball player shooting a freethrow releases at the correct angle and follows through with the help of computer vision

The Shotform Corrector is an advanced computer vision application that aims to accurately determine if a basketball player shooting a freethrow releases within an optimum angle range. Proper shooting form is crucial for young and developing players. The project leverages the power of Python and various computer vision techniques to allow players to correct their shot form in real time and develop good shooting form without the need of a coach or teamate present. 

![corrector-demo](https://github.com/AggieSportsAnalytics/ShotFormCorrecter/blob/main/demo.gif?raw=true)

# 🔑 Key Features

## Shooter Release Angle Measurement

The project employs a technique known as pose estimation inorder to measure the elbow angle of a basketball player when releasing a free throw.

![Screenshot 2023-08-29 at 3 30 41 PM](https://github.com/AggieSportsAnalytics/ShotFormCorrecter/blob/main/Screenshot%202023-09-30%20at%206.56.31%20PM.png?raw=true)
**_<br>Each small circle represents a joint being used for release angle measurement. Here, the angle is being measured by finding the angle made between the shoulder, elbow, and pinky_**

### 💻 Code

For pose estimation we utilized MediaPipe's pretrained models. MediaPipe allowed us to choose which joints to detect and measure the angle between them. We then used OpenCv to visually mark the joints being used and display the angle measurement.

```py
import cv2
import mediapipe as mp
import time
import math


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 300, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
```

## Basketball detection

Inorder to detect when the player is in a shooting motion, we check to see if the ball is above the player's shoulder. At that point we start measuring the angle between the three chosen joints. Inorder to do this, we need to detect and track the basketball as well. We utilized a pretrained Yolo v8 model for basketball detection. 

### 💻 Code

```py
class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("runs/detect/train/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']
```

```py
results = self.model(self.frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            current_class = self.class_names[cls]

            center = (int(x1 + w / 2), int(y1 + h / 2))

            # Only create ball points if high confidence or near hoop
            if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                self.ball_pos.append((center, self.frame_count, w, h, conf))
                cvzone.cornerRect(self.frame, (x1, y1, w, h))

            # Create hoop points if high confidence
            if conf > .5 and current_class == "Basketball Hoop":
                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                cvzone.cornerRect(self.frame, (x1, y1, w, h))
```

We also need to track the ball inorder to detect if the player has followed through or not. We do this by checking if after the ball has left the players hand, the angle between the three joints being measured is around 180 degrees, implying they have followed through with their shooting hand. 

```py
release_angle, follow_through_angle, left_feet_heights, right_feet_heights = self.run()
```

```py
def run(self):
    release_angle = None
    follow_through_angle = -1
    left_feet_heights = []
    right_feet_heights = []
    last_angle = 0
    num_frame = 0
    angle_decreasing = False
    already_beeped = False
    release_frame = -1
    already_released = False
    follow_through_iter = 10
    follow_through_best = 155
    did_follow_through = False

    frames = []
    while True:
        ret, self.frame = self.cap.read()
        if self.frame is None:
            break
        self.img = self.detector.findPose(self.frame, draw=False)
        self.lmList = self.detector.findPosition(self.img, draw=False)

        self.detector.findAngle(self.img, 12, 14, 16, draw=True)

        if not ret:
            # End of the video or an error occurred
            break

        results = self.model(self.frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                center = (int(x1 + w / 2), int(y1 + h / 2))

                # Only create ball points if high confidence or near hoop
                if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

                # Create hoop points if high confidence
                if conf > .5 and current_class == "Basketball Hoop":
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

        cv2.circle(self.img, (self.lmList[29][1], self.lmList[29][2]), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(self.img, (self.lmList[30][1], self.lmList[30][2]), 10, (0, 0, 255), cv2.FILLED)

        left_feet_heights.append(self.lmList[29][2])
        right_feet_heights.append(self.lmList[30][2])

        if did_follow_through:
            if follow_through_angle > follow_through_best:
                cv2.putText(self.frame, 'GOOD FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(self.frame, 'DIDN\'T FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        elif already_released and not did_follow_through:
            if self.frame_count - release_frame > follow_through_iter:
                angle = self.detector.findAngle(self.img, 12, 14, 16, draw=True)

                if angle > follow_through_angle:
                    follow_through_angle = angle

                # cv2.imwrite(f'follow-through/{temp}.jpg', self.img)
                follow_through_iter += 1
                # print(f'{temp} error -> {follow_through_angle} angle')

                if follow_through_iter >= 40:
                    did_follow_through = True

            if self.frame_count - release_frame < 12:
                cv2.putText(self.frame, 'REMEMBER TO FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        elif not did_follow_through:
            if(self.ball_pos):
                bx, by = self.ball_pos[-1][0][0], self.ball_pos[-1][0][1] #most recent coordinates for ball
                sx, sy = self.lmList[12][1], self.lmList[12][2] #shoulder coordinates

                angle_error = 15
                error_margin = 50
                dist_margin = 125

                if(sy < by):
                    self.below_shoulder = True
                elif self.below_shoulder and sy > by + error_margin:
                    angle = self.detector.findAngle(self.img, 12, 14, 16, draw=True)

                    if angle < last_angle:
                            angle_decreasing = True
                    else:
                        angle_decreasing = False

                    if self.frame_count % 10 == 0:
                        last_angle = angle

                    if angle_decreasing and abs(angle - 60) <= angle_error:
                        print(f'loaded angle: {angle}')
                        cv2.putText(self.frame, 'RELEASE NOW', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                        if not already_beeped:
                            subprocess.run('osascript -e "beep"', shell=True)
                            already_beeped = True

                        # cv2.imwrite('loaded_angle.jpg', self.img)

                    self.above_shoulder = True

                    rpx, rpy = self.lmList[12][1], self.lmList[12][2] #right pinky

                    dist = math.dist((rpx, rpy), (bx, by))

                    if dist > dist_margin:
                        release_angle = angle
                        # cv2.imwrite(f'release-points/RP-{os.path.splitext(fp)[0]}.jpg', self.img)
                        cv2.putText(self.frame, 'REMEMBER TO FOLLOW THROUGH', (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                        # cv2.imwrite(f'test.jpg', self.img)
                        release_frame = self.frame_count
                        already_released=True
```

# 🪴 Areas of Improvement

- Accuracy: Currently the ball detection and pose estimation work well when the inputted video has ideal conditons. We need to improve the models we're using to allow for more variation in video quaulity
- Ideal release range: The ideal angle at which a free throw is shot is dependent on height which we aren't considering currently. We could improve the utility of the project by taking into account the height of the player as well.

# 🚀 Further Uses

- Expanding to all shots: Currently we only use video footage of free throws because there are less variables to deal with. However, we could improve the utility of our project by also correcting shot form where multiple people are present, such as in a real game setting.

# 💻 Technology

- OpenCV
- YoloV8
- MediaPipe
