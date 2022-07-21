# encoding:utf-8
import cv2  # python中的opencv-python库，著名的计算机视觉库
import numpy as np
#采集视频、显示并保存        
dictionary = {0:'a',1:'b',2:'c'}
import tensorflow as tf
from tensorflow import keras
model = tf.keras.models.load_model('asl_model')

def capturevideoSave(imgName,camera = 0):
    # camera 表示摄像头，内置摄像头:0, 外接摄像为按照顺序依次为1,2,3...
    cap=cv2.VideoCapture(camera)
    global_num = 0 
    while(1):
        ret,frame = cap.read()
        #镜像处理显示
        frame = np.fliplr(frame)
        #图片压缩到合适大小
        frame = cv2.resize(frame,(200,200))
        cv2.imshow("capture", frame)
        frame = frame/225
        prediction = model(frame)
        print(dictionary[np.argmax(prediction)])
        #更改窗口大小
        #cv2.resizeWindow("capture", 640, 480)
        k=cv2.waitKey(1) #等待1ms，获取用户的键盘输入
        
        
        if k==ord(' '): #如果用户输入空格，将当前帧用imgName作为文件名保存
            cv2.imwrite(imgName,frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return imgName  #返回保存的文件名

# 模块测试成功后，请将test设置为False，在项目程序中避免重复调用
test=True
#test=False
if test:
    capturevideoSave("./tmp.jpg",0)

    