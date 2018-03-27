# -*- coding:utf-8 -*-
import cv2
import os
for file in os.listdir("out/"):
    filename = "out/"+file
    img = cv2.imread(filename)
    res = cv2.resize(img,(720,360))
    outpath = "images/"+file
    cv2.imwrite(outpath,res)