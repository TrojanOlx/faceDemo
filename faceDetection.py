import cv2.cv2 as cv2
import glob as gb
import opencvContrast as ct


face_detection =cv2.CascadeClassifier("./trained_models/detection_models/haarcascade_frontalface_default.xml")


camera=cv2.VideoCapture(0)

while True:
    ret, frame = camera.read() # 逐帧采集视频流
    if not ret:
        continue
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces=face_detection.detectMultiScale(gray_image,1.3,5)
    for face_coordinates in faces:
        x, y, width, height = face_coordinates
        cv2.rectangle(frame, (x ,y ), (x + width , y + height), (255,0,0), 2)
        

        # 将头像保存到 images 文件夹下面
        # 可以多收集几个人的
        # cv2.imwrite("./images/"+str(x)+".jpg")


        cropImg = frame[y:y+height,x:x+width] # 截取头像
        ss = ct.contrast(cropImg) #进行对比
        cv2.putText(frame, ss, 
                    (x+20, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
