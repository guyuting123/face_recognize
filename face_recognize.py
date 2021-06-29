import os
import cv2
import numpy as np
import utils.utils as utils
from net.inception import InceptionResNetV1
from net.mtcnn import mtcnn

# 上面解释，下面是具体的代码
# 创建人脸检测类face_rec(),其中包含两个函数:init、recognize
class face_rec():
    def __init__(self): #在创建的时候就会运行
        #-------------------------#
        #   1、创建mtcnn的模型（）
        #   用于检测人脸
        #-------------------------#
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.6,0.8]#使用了3个神经网络，0.5-Pnet，0.6-Rnet，0.8-Onet（三个网络的门限）
               
        #-----------------------------------#
        #   2、载入facenet
        #   将检测到的人脸转化为128维的特征向量
        #-----------------------------------#
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        #-----------------------------------------------#
        #  3、创建人脸识别仓库，对数据库中的人脸进行编码
        #   创建两个列表（known_face_encodings、known_face_names）
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset")#找出文件face_dataset下的图片
        self.known_face_encodings=[]
        self.known_face_names=[]
        for face in face_list:  #对face_dataset里的所有人脸进行遍历，
            name = face.split(".")[0]
            img = cv2.imread("./face_dataset/"+face)    #读取人脸图片
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #将图片从BGR转成RGB
            #---------------------#
            #   利用mtcnn进行人脸识别
            #---------------------#
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)
            #---------------------#
            # facenet要传入一个160x160的图片，所以要转换成正方形
            #---------------------#
            rectangles = utils.rect2square(np.array(rectangles))
            #-----------------------------------------------#
            #   将转换后的图片读取出来
            rectangle = rectangles[0]
            #   利用landmark对人脸进行矫正
            #-----------------------------------------------#
            # landmark会生成5个标记点，这5个标记点可以帮助我们将图片对齐摆正，使得歪着的图片也有利于识别
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            # 1、将图片截取下来
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

            # 2、利用5个标记点进行对齐
            crop_img, _ = utils.Alignment_1(crop_img,landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)
            #--------------------------------------------------------------------#
            #   将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取（得到人脸编码face_encoding）
            #--------------------------------------------------------------------#
            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)
            # 将128维特征向量存入到已知的编码和名字列表中
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
    # 此前：所有操作便可得到人脸仓库face_dataset中的所有人脸的128维特征向量即人脸编码
    # 此后：将检测到的人脸与仓库中的人脸编码进行匹配

    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)#获取图片的高和宽
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)#将图片转成RPG

        #--------------------------------#
        #   检测人脸
        #--------------------------------#
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)#利用mtcnn获取图片中所有的人脸

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)
        rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            #---------------#
            #   截取图像
            #---------------#
            landmark = np.reshape(rectangle[5:15], (5,2)) - np.array([int(rectangle[0]), int(rectangle[1])])#进行人脸矫正
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #-----------------------------------------------#
            #   利用人脸关键点进行人脸对齐
            #-----------------------------------------------#
            crop_img,_ = utils.Alignment_1(crop_img,landmark)
            crop_img = np.expand_dims(cv2.resize(crop_img, (160, 160)), 0)

            face_encoding = utils.calc_128_vec(self.facenet_model, crop_img)#将矫正后的图片传入到facenet的模型中
            face_encodings.append(face_encoding)#将提取出的128维特征向量保存到此列表中
################            此上可以将实时的人脸进行矫正，并将其128维的特征向量进行提取           #############################

        face_names = []

        for face_encoding in face_encodings:
            #-------------------------------------------------------#
            #   将实时监测得到的人脸编码与数据库中所有的人脸进行对比（距离），计算得分
            #-------------------------------------------------------#
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.9)#容忍度:如果距离<0.9，便可判定两张脸相似
            name = "Unknown"
            #-------------------------------------------------------#
            #   找出距离最近的人脸
            #-------------------------------------------------------#
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            #-------------------------------------------------------#
            #   取出这个最近人脸的评分
            #-------------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)#将找到的距离最近的脸的名字保存到face_names列表中

        rectangles = rectangles[:,0:4]#取出方框
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 
        return draw

if __name__ == "__main__":
    dududu = face_rec()
    video_capture = cv2.VideoCapture(0) #1、读取摄像头

    while True:
        ret, draw = video_capture.read()#2、读取图片
        dududu.recognize(draw) #recognize：是将检测的图片进行显示，是所用两个函数中的其中一个
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
