# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos
import PoseEstimationMin as pm
import os
import subprocess


class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("runs/detect/train/weights/best.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']

        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture('practice-videos/D9_1_449_detail.mp4')

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.detector = pm.poseDetector()
        self.img = None
        self.lmList = []

        self.below_shoulder = False
        self.above_shoulder = False

        release_angle, follow_through_angle, left_feet_heights, right_feet_heights = self.run()

        print(f'RELEASE ANGLE: {release_angle}')
        print(f'FOLLOW THROUGH ANGLE: {follow_through_angle}')

        left_release_height = np.mean(left_feet_heights[0:50]) - left_feet_heights[-1]
        right_release_height = np.mean(right_feet_heights[0:50]) - right_feet_heights[-1]
        print(f'RELEASE HEIGHT: {np.mean([left_release_height, right_release_height])}')


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

        while True:
            ret, self.frame = self.cap.read()
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

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()

        if release_angle > 180:
            release_angle = 360 - 180

        return release_angle, follow_through_angle, left_feet_heights, right_feet_heights

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

    def display_score(self):
        # Add text
        # text = str(self.makes) + " / " + str(self.attempts)
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        # cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    # for fp in os.listdir('practice-videos'):
    #     ShotDetector(f'practice-videos/{fp}')
    ShotDetector()