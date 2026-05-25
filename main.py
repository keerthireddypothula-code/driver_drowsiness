from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
import numpy as np

# Sound parameters
FREQUENCY = 2500
DURATION = 1000

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])  # Vertical distance
    B = dist.euclidean(mouth[0], mouth[6])  # Horizontal distance
    mar = A / B
    return mar

# Thresholds and counters
EYE_AR_THRESH = 0.25
EYE_AR_SLEEP_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 30
SLEEP_CONSEC_FRAMES = 50
MAR_THRESH = 0.5
YAWN_CONSEC_FRAMES = 20

COUNTER_EYE = 0
COUNTER_YAWN = 0
COUNTER_SLEEP = 0

# Full path to the shape predictor file (make sure it's there!)
SHAPE_PREDICTOR = r"C:\Users\ASUS\Downloads\DRIVER DROWSINESS\DRIVER DROWSINESS WITH MOUTH\shape_predictor_68_face_landmarks.dat"

# Initialize video capture and models
cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

# Get facial landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[lStart:lEnd]
        leftEye = shape[leftEyeStart:leftEyeEnd]
        rightEye = shape[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Sleep detection
        if ear < EYE_AR_SLEEP_THRESH:
            COUNTER_SLEEP += 1
            if COUNTER_SLEEP >= SLEEP_CONSEC_FRAMES:
                cv2.putText(frame, "SLEEP DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(FREQUENCY, DURATION)
        else:
            COUNTER_SLEEP = 0

        # Drowsiness detection
        if ear < EYE_AR_THRESH and ear >= EYE_AR_SLEEP_THRESH:
            COUNTER_EYE += 1
            if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(FREQUENCY, DURATION)
        else:
            COUNTER_EYE = 0

        # Yawn detection
        if mar > MAR_THRESH:
            COUNTER_YAWN += 1
            if COUNTER_YAWN >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN DETECTED!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(FREQUENCY, DURATION)
        else:
            COUNTER_YAWN = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
