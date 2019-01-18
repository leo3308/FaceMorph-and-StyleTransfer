from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from faceMorph import *
from face_landmark_detection import face_landmark_detection
from eval import *
import cv2
import numpy as np
from PIL import Image,ImageDraw
import dlib
import sys
import threading
import time
import random


def getTranferStyle():
    style_list = ["cubist.ckpt-done", "denoised_starry.ckpt-done", "feathers.ckpt-done", "mosaic.ckpt-done", "scream.ckpt-done", "udnie.ckpt-done", "wave.ckpt-done"]
    random.seed(time.time())
    index = random.randint(0,6)
    path_to_style = "/Users/shuyucho/Desktop/碩一上/ICG/hw/final/style/" + style_list[index]
    return path_to_style


def getFileName(input_str):
    file = input_str.split("/")
    return file[-1]

class MyThread(threading.Thread):
    def run(self):
        self.threadLife = True
        while self.threadLife:
            widget.update()
            time.sleep(0.1)
    def stop(self):
        self.threadLife = False

class MyWidget(QWidget):
    def __init__(self, parent = None):
        super(MyWidget, self).__init__(parent)
        self.createLayout()

    def createLayout(self):
        self.filename1 = ""
        self.filename2 = ""
        
        self.select_image_btn1 = QPushButton("Select Image 1")
        self.select_image_btn1.clicked.connect(self.getImage)
        self.image1_output = QLabel()
        self.image1_output.setFixedSize(240,320)
        self.select_image_btn2 = QPushButton("Select Image 2")
        self.select_image_btn2.clicked.connect(self.getImage)
        self.image2_output = QLabel()
        self.image2_output.setFixedSize(240,320)
        
        self.textOfHint = QLabel("Alpha Value")
        self.alpha_value = QLineEdit()
        self.alpha_value.setFixedWidth(50)
        self.morphing_btn = QPushButton("Morphing!!")
        self.morphing_btn.clicked.connect(self.getMorphImage)
        self.morphing_output = QLabel()
        self.morphing_output.setFixedSize(240,320)
        self.original_pic = QPushButton("Original")
        self.original_pic.clicked.connect(self.getOriginalMorphPic)
        self.style_pic = QPushButton("Change Style")
        self.style_pic.clicked.connect(self.getStylePic)
        
        self.gif_btn = QPushButton("Generate GIF")
        self.gif_btn.clicked.connect(self.getMorphGIF)
        self.morph_gif = QLabel()
        self.morph_gif.setFixedSize(240,320)
        
        
        layout = QVBoxLayout()
        layout.addWidget(self.select_image_btn1)
        layout.addWidget(self.image1_output)
        
        layout2 = QVBoxLayout()
        layout2.addWidget(self.select_image_btn2)
        layout2.addWidget(self.image2_output)
        
        txtLayout = QHBoxLayout()
        txtLayout.addWidget(self.textOfHint)
        txtLayout.addWidget(self.alpha_value)
        txtLayout.addWidget(self.morphing_btn)
        change_style_layout = QHBoxLayout()
        change_style_layout.addWidget(self.original_pic)
        change_style_layout.addWidget(self.style_pic)
        morph_layout = QVBoxLayout()
        morph_layout.addLayout(txtLayout)
        morph_layout.addWidget(self.morphing_output)
        morph_layout.addLayout(change_style_layout)
        
        gifLayout = QVBoxLayout()
        gifLayout.addWidget(self.gif_btn)
        gifLayout.addWidget(self.morph_gif)
        
        
        hbox = QHBoxLayout()
        hbox.addLayout(layout)
        hbox.addLayout(morph_layout)
        hbox.addLayout(layout2)
        hbox.addLayout(gifLayout)
        self.setLayout(hbox)
    
    def getImage(self):
        sender = self.sender()
        image = QFileDialog.getOpenFileName(self, "Open file","", "Image files(*.jpg *.png *)")
        imagePath = image[0]
        if sender is self.select_image_btn1:
            scaledImg = QPixmap(imagePath).scaled(self.image1_output.width(), self.image1_output.height())
            self.image1_output.setPixmap(scaledImg)
            self.filename1 = getFileName(imagePath)
        elif sender is self.select_image_btn2:
            scaledImg = QPixmap(imagePath).scaled(self.image2_output.width(), self.image2_output.height())
            self.image2_output.setPixmap(scaledImg)
            self.filename2 = getFileName(imagePath)
    def getMorphImage(self):
        alpha = float(self.alpha_value.text())
        face_landmark_detection(self.filename1)
        face_landmark_detection(self.filename2)
        imgMorph = MorphImage(alpha,self.filename1,self.filename2)
        #save morphed image
        img_array = imgMorph[:,:,[2,1,0]]
        img = Image.fromarray(img_array.astype("uint8"), "RGB")
        img.save("morphed_image.jpg","JPEG")
        
        imgMorph = np.uint8(imgMorph)
        imgMorph2 = cv2.cvtColor(imgMorph,cv2.COLOR_BGR2RGB)
        height, width, channels= imgMorph2.shape
        bytesPerLine = channels * width
        qImg = QImage(imgMorph2.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap01 = QPixmap.fromImage(qImg)
        pixmap_image = QPixmap(pixmap01).scaled(self.morphing_output.width(), self.morphing_output.height())
        self.morphing_output.setPixmap(pixmap_image)
    def getMorphGIF(self):
        images = []
        for i in range(10):
            alpha = i/10 + 0.1
            img_array = MorphImage(alpha,self.filename1,self.filename2)
            img_array = img_array[:,:,[2,1,0]]
            img = Image.fromarray(img_array.astype("uint8"), "RGB")
            img = img.resize((240,320), Image.ANTIALIAS)
            images.append(img)
        images[0].save('test.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        movie = QMovie("test.gif")
        self.morph_gif.setMovie(movie)
        movie.start()
    def getOriginalMorphPic(self):
        scaledImg = QPixmap("morphed_image.jpg").scaled(self.morphing_output.width(), self.morphing_output.height())
        self.morphing_output.setPixmap(scaledImg)
    def getStylePic(self):
        path_to_style = getTranferStyle()
        style_transfer(path_to_style, "morphed_image.jpg")
        scaledImg = QPixmap("/Users/shuyucho/Desktop/碩一上/ICG/hw/final/generated/res.jpg").scaled(self.morphing_output.width(), self.morphing_output.height())
        self.morphing_output.setPixmap(scaledImg)



app = QApplication(sys.argv)

widget = MyWidget()
t = MyThread()
t.setDaemon(True)
t.start()
widget.setWindowTitle("NTU-ICG Final Project")
widget.show()

app.exec_()
t.stop()
t.join()

