# -*- coding:utf-8 -*-
import cv2
VIDEO_NAME = "video_data/02.mp4"
video = cv2.VideoCapture(VIDEO_NAME)
if video.isOpened() == False:  # 异常检测
    print("Open Video Error!")
    exit()  # 退出
framecount = video.get(cv2.CAP_PROP_FRAME_COUNT)
outpath = "out/"
for i in range(int(framecount)):

    success, frame = video.read()
    if i%5 == 0:
        cv2.imshow("frame",frame)
        res = cv2.resize(frame, (1280, 720))
        cv2.imwrite(outpath+str((i+5)//5).zfill(4)+".jpg",res)
    if cv2.waitKey(10) & 0xff == ord('q'):
        exit()

cv2.destroyAllWindows()





