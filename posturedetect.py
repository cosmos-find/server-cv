import cv2
from pathlib import Path



def production(frame):
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
    # 각 파일 path
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose_iter_160000.caffemodel"
    
    # 위의 path에 있는 network 모델 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inputWidth=320;
    inputHeight=240;
    inputScale=1.0/255;
    
    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    
    imgb=cv2.dnn.imagesFromBlob(inpBlob)
    # cv2.imshow("motion",(imgb[0]*255.0).astype(np.uint8))
    
    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()


    # 키포인트 검출시 이미지에 그려줌
    points = []

    def turtleNeck(num1, num2):
        if num1 and num2:
            if abs(num1[0] - num2[0])>= 10:
                print("error 목 넣으세여")
                return 10

    def unbalanceArm(num1, num2):
        if num1 and num2:
            if abs(num1[1] - num2[1])>= 30:
                print("팔일자!!")
                

    for i in range(0,15):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
    
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.1 :    
            points.append((int(x), int(y)))
        else :
            points.append(None)
    

    statement = turtleNeck(points[0],points[1])
    if statement == 10:
        neck = "Turtle Neck"
        return neck

