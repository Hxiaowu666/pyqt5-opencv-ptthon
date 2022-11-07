import sys
import time
from PyQt5.QtCore import*
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import cv2 as cv
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.init_ui()
        self.timer_camera = QtCore.QTimer()     #定义定时器，用于控制显示视频的帧率
        self.cap = cv.VideoCapture(0)           #获取视频流
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()

    def init_ui(self):
        self.ui = uic.loadUi('E:\\yan\\test4_faceDetect\\QT练习\\ui\\FaceDetect.ui')# 初始化ui
        print(self.ui.__dict__)  # 获取ui对象的所有属性
        # 绑定对象
        self.FaceCollect = self.ui.FaceCollectButton  # 人脸采集
        self.FaceTrain = self.ui.FaceTrainButton  # 人脸训练
        self.FaceDetect = self.ui.FaceDetectButton  # 人脸识别
        self.outputText = self.ui.textBrowser  # 文本显示
        self.outputText.setPlaceholderText('输出信息')# 输出文本框提示符
        self.CloseCameraButton = self.ui.Colse_camera_button #关闭摄像头按钮
        self.Show_camera = self.ui.Camera        #摄像头显示
        self.Show_camera.setStyleSheet("QLabel{background-color:rgb(50,50,50);}")# 设置lable背景颜色
        #self.Show_camera.setScaledContents(True)  # 让图片自适应label大小
        self.RecordTime = self.ui.RecordTimeButton  #打卡
        #绑定槽函数
        self.FaceCollect.clicked.connect(self.Face_Collect)
        self.FaceTrain.clicked.connect(self.Face_Train)
        self.FaceDetect.clicked.connect(self.Face_Detect)
        self.CloseCameraButton.clicked.connect(self.close_camera)
        self.RecordTime.clicked.connect(self.daka)

    def printf(self, mes):
        self.ui.textBrowser.append(mes)  # 在指定的区域显示提示信息
        self.cursot = self.ui.textBrowser.textCursor()
        self.ui.textBrowser.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents()

    def show_camera(self):
        if self.timer_camera.isActive() == False:   #如果定时器未启动
            flag = self.cap.open(0)
            if flag == False:
                return
            else:
                self.timer_camera.start(30)     #开启定时器
        if self.cap:
            self.flag, self.image = self.cap.read()  # 获取帧图片
            self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

    def close_camera(self):
        self.timer_camera.stop()
        self.Show_camera.clear()
        self.cap.release()

    # 汉字转换函数
    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "chinese.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

    # 人脸检测函数
    def face_detect(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为灰度图片，简化矩阵、提高运算速度
        # 调用人脸检测的级联分类器
        face_classifier = cv.CascadeClassifier(
            'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
        # 对人脸进行检测，每次图像缩小的比例为1.1，每一个目标至少检测5次
        face_feature = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face_feature:  # 遍历x个人脸
            cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)  # 对人脸画框
        if len(face_feature) > 1:
            self.image = self.cv2AddChineseText(img, '检测到多人', (0,30), (0, 255, 0), 30)
        elif len(face_feature) == 1:
            if face_feature[0][2] >300 or face_feature[0][3] > 300:
                self.image = self.cv2AddChineseText(img, '太近了', (0, 50), (0, 255, 0), 30)
            if face_feature[0][2] <150 or face_feature[0][3] < 150:
                self.image = self.cv2AddChineseText(img, '太远了', (0, 50), (0, 255, 0), 30)
            else:
                if self.num <= 20:
                    imgFile = 'E:\\yan\\test4_faceDetect\\QT练习\\savePhotos\\' + self.name + '.' + str(int(
                        self.existPerson[0]) + 1) + '.' + str(self.num) + '.jpg'
                    cv.imencode('.jpg', self.image)[1].tofile(imgFile)  # 存储中文图片
                    print('已完成:{:.2%}'.format(self.num/20))   #显示存储进度
                    MyWindow.printf(self,'已完成:{:.2%}'.format(self.num/20))
                    self.num += 1
        show = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.Show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    #人脸采集
    def Face_Collect(self):
        # 读取文件信息
        self.name,N = QInputDialog.getText(self, '用户输入', '名字')
        print(self.name)
        fr = open('data.txt', 'r',encoding='utf-8')
        self.existPerson = fr.readlines()
        print(self.existPerson)  # 已存储   人数  姓名[]
        fr.close()
        self.num = 1
        self.cap.open(0)
        self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
        while self.cap.isOpened():
            self.flag, self.image = self.cap.read()  # 获取帧图片
            cv.putText(self.image, str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))), (0, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.face_detect(self.image)
            cv.waitKey(1)
            if self.num == 21:
                self.existPerson[0] = (str(int(self.existPerson[0]) + 1) + '\n')
                self.existPerson.append(self.name + '\n')
                fw = open('data.txt', 'w',encoding='UTF-8')
                fw.writelines(self.existPerson)  # 更新文件信息
                fw.close()
                print('照片已存储\n')
                MyWindow.printf(self,'照片已存储')
                self.num += 1

    #保存人脸信息
    def saveFacefunc(self,path):
        faceSample = []  # 存储人脸数据
        faceName = []  # 存储人脸姓名
        imagePath = [os.path.join(path, f) for f in os.listdir(path)]  # 获取到所有目标文件的完整路径
        face_detect = cv.CascadeClassifier(
            'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')  # 调用人脸检测的级联分类器
        for imagepath in imagePath:
            PIL_img = Image.open(imagepath).convert('L')  # 打开该图片，L表示转化为灰度图像 ---简化矩阵、加快运算速度
            img_numpy = np.array(PIL_img, 'uint8')  # 将图像数据转换为数组
            faces = face_detect.detectMultiScale(img_numpy)  # 获取图片的人脸特征
            id = int(os.path.split(imagepath)[1].split('.')[1])  # 仅获取序号
            for x, y, w, h in faces:
                faceName.append(id)
                faceSample.append(img_numpy[y:y + h, x:x + w])  # numpy数组切片，从y取到y+h行，从x取到x+w列，构成新的数组，把所画的方框放入列表中
        return faceName, faceSample
    #人脸训练
    def Face_Train(self):
        faceName, faceSample = self.saveFacefunc(path='.//savePhotos')  # 获取姓名和脸部特征
        # 采用LBPH算法
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(faceSample, np.array(faceName))  # 训练
        # 保存训练好的文件
        recognizer.write('trainer.yml')
        print('文件已保存')
        MyWindow.printf(self,'文件已保存')

    # 人脸识别函数
    def face_recongnition(self, img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转为灰度图
        facedetect = cv.CascadeClassifier(
            'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')  # 调用分类器
        face = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # 检测到5次 成功认定
        for x, y, w, h in face:
            cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)  # 画框
            self.lable, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])  # 返回标签和置信度
            MyWindow.printf(self,self.ExistPerson[self.lable])
            if confidence >= 60:  # 置信度大于50，返回unknow
                cv.putText(img, 'unknow', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                img = self.cv2AddChineseText(img, self.ExistPerson[self.lable], (x + 10, y - 30), (0, 255, 0), 30)
                # cv.putText(img,str(names[lable-1]),(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0),1)
                cv.putText(img, (str(round(confidence, 3))), (x + 125, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                           (255, 0, 0), 1)
        show = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.Show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def daka(self):
        print(self.ExistPerson[self.lable] + '已打卡!  ' +time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        MyWindow.printf(self,self.ExistPerson[self.lable] + '已打卡!  ' +time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        f = open('record.txt', 'a')
        f.write(self.ExistPerson[self.lable] + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        f.close()

    def Face_Detect(self):
        # 导入已经训练完成的模型
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer.yml')
        # 读取文件信息
        fr = open('data.txt', 'r',encoding='utf-8')
        self.ExistPerson = fr.readlines()
        fr.close()
        # 打开摄像头
        self.cap.open(0)
        while self.cap.isOpened():
            cv.waitKey(1)       #不加waitkey  当未检测到人脸时卡顿？
            self.flag, self.image = self.cap.read()  # 获取帧图片
            cv.putText(self.image, str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))), (0, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.face_recongnition(self.image)  # 传入图像进行人脸识别

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.ui.show()
    app.exec()