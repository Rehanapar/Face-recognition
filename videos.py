import cv2

trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture('video/video.mp4')

while True:
    success, frame = video.read()

    if success == True:

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = trainedDataset.detectMultiScale(gray_image)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('video', frame)
        key = cv2.waitKey(1)
       

    else:

        print("Video Conmpleted")
        break