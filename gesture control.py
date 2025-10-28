import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pycaw.pycaw as pycaw

class Config:
    MOVE_THRESHOLD_RESPONSE = 25e-6
    MOVE_THRESHOLD_NUMBER = 10
    

class FindMove:

    def __init__(self):
        self.detector = cv.ORB_create()

    def detect(self, frame):
        
        move = False
        kp, _ = self.detector.detectAndCompute(frame, None)
        kp_len = sum(True for k in kp if k.response < Config.MOVE_THRESHOLD_RESPONSE)
        if (kp_len > Config.MOVE_THRESHOLD_NUMBER):
            move = True
        return move


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Камера недоступна')

    findMove = FindMove()

    while cap.isOpened():
        ret, frame = cap.read()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if not ret:
            break
        frame = cv.resize(frame, (448, 448))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if findMove.detect(gray):
            # обнаруживаем жест
            cv.imshow('gray', gray)
        else:
            cv.imshow('gray', frame)


        cv.imshow('frame', frame)
    cap.release()
                
    cv.destroyAllWindows()
