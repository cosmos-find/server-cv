import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import datetime as dt
import time

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    # scipy의 distance 모듈을 이용하여 두 지점 사이의 거리를 반환한다.
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


def sleepdetect(frame, EYE_CLOSED_COUNTER):
    FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
    MINIMUM_EAR = 0.2
    # 임계값은 0.2로 설정
    MAXIMUM_FRAME_COUNT = 10

    faceDetector = dlib.get_frontal_face_detector()
    landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    (leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    image = imutils.resize(frame, width=800)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceDetector(grayImage, 0)

    for face in faces:
        faceLandmarks = landmarkFinder(grayImage, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

    
    if EYE_CLOSED_COUNTER >= 5:
        print("sleep statement")
        eye = "eye-closed"
        return eye

    if ear < MINIMUM_EAR:
        return 100



# recognition.py의 작동원리 !
# shape_predictor_68_face_landmarks.dat이라는 모델을 이용 --> 얼굴의 68개 지점의 좌표를 뽑아냄.
# dlib 모듈 이용(shape_predictor 함수 적용)
# 1. lefteye와 righteye의 눈이 감긴 정도를 파악. eye_aspect_ratio 함수를 이용
# 2. ear --> 왼쪽과 오른쪽 눈이 감긴 정도를 나타냄. (기하학적 방법 이용 --> 눈의 6개 지점을 빼냄)
# 3. 왼쪽 눈과 오른쪽 눈이 감긴 정도를 평균 내어서 이용할 특정 값을 도출
# 4. 특정 값이 임계점을 넘으면 졸고 있다고 판단한다.
# 5. 졸고 있다고 판단되기 시작한 후, 졸고 있는 시간이 길어지면 sleep statement로 파악한다. 