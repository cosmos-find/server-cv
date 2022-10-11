import cv2
import numpy as np
from os import makedirs
from os import listdir
from os.path import isdir, isfile, join
import pandas as pd
import os

# haarcascade_frontalface_default.xml을 이용하여 사진에서 얼굴을 검출해낸다.

# 얼굴 검출 함수
def face_extractor(img):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴!
    return cropped_face

def take_pictures(name):
    
    face_dirs = 'faces/'
    # 해당 이름의 폴더가 없다면 생성
    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)

    # 카메라 ON    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 
        if face_extractor(frame) is not None:
            
            count+=1
            face = cv2.resize(face_extractor(frame),(250,250))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = face_dirs + name + '/user'+str(count)+'.jpg'
            print(file_name_path)
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass
        
        if cv2.waitKey(1)==13 or count==40:
            break

    cap.release()
    print('Colleting Samples Complete!!!')

if __name__ == "__main__":
    take_pictures('someone')