# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:04:57 2021

@author: Admin
"""

import sys
import cv2
import os
from PyQt5 import QtCore, QtGui, QtWidgets

from main_window_ui import Ui_MainWindow
import numpy as np



class ImaqArt(Ui_MainWindow):
    
    
    def __init__(self, dialog):
        self.n = 0
        Ui_MainWindow.__init__(self)
        self.setupUi(dialog)
        self.Browse.clicked.connect(self.browse_files)
        self.EqualisationBtn.clicked.connect(self.histogram_equilisation)
        self.LogTransformBtn.clicked.connect(self.log_transformation)
        self.SaveImageBtn.clicked.connect(self.save_image)
        self.UndoBtn.clicked.connect(self.undo)
        self.UndoAllBtn.clicked.connect(self.undo_all)
        # self.demo_list.itemClicked.connect(self.display_selected_item)
        # self.camera.clicked.connect(self.run_camera)
        self.blur_slider.sliderReleased.connect(self.blur)
        self.gamma_slider.sliderReleased.connect(self.gamma_correction)
        self.sharpen_slider.sliderReleased.connect(self.sharpen)


    def blur(self):
        print('image blurring in process...')

        img = self.image
        bi = self.blur_slider.value()           #blur index

        print(bi)
        
        size = len(np.shape(img))
        if size == 3:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_temp = img_hsv[:,:,2]
    
        elif size == 2:
            img_temp = img
        
        h,w = np.shape(img_temp)
        print(h,w)
            
        img_temp_blur = np.zeros(shape=(h,w))
        img_temp = img_temp.astype(int)
        
        for i in range (bi, h-bi):
            for j in range (bi, w-bi):
                s = 0
                for k in range(-bi,bi+1):
                    for l in range (-bi, bi+1):
                        s = s + img_temp[i+k][j+l]
                img_temp_blur[i][j] = s//((2*(bi)+1)**2)
        
        img_temp_blur = img_temp_blur.astype(np.uint8)

        
        if size == 3:
            img_blur_hsv = np.dstack(tup = (img_hsv[:,:,0], img_hsv[:,:,1], img_temp_blur))
            img_blur = cv2.cvtColor(img_blur_hsv, cv2.COLOR_HSV2RGB)

            
        elif size == 2:
            img_blur = img_temp_blur
        
        
       # img = cv2.blur(img,(val,val)) 
        self.display_image(img_blur)
        print('image blurring in done')

        
    def browse_files(self):
        
        filter = "Image files (*.jpg *.gif *.png)"
        filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file',os.getcwd() ,filter)        
        self.filename = filename[0]

        # self.image_filename.setText(self.filename)
        self.display_file(self.filename)
        img = cv2.imread(self.filename)
        self.org_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(np.shape(img))
 
    
    def display_file(self, filename):
        img = cv2.imread(self.filename)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixmap = QtGui.QPixmap(filename)
        self.DisplayImage.setScaledContents(True)
        self.DisplayImage.setPixmap(pixmap)
 
    
    def display_image(self, img):
        
        self.last_image = self.image
        self.image = img
        h, w, ch = img.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
        self.DisplayImage.setPixmap(QtGui.QPixmap.fromImage(p))
        


    def histogram_equilisation(self):
        img = self.image
        
        size = len(np.shape(img))
        if size == 3:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_temp = img_hsv[:,:,2]
            print
    
        elif size == 2:
            img_temp = img
            
        temp_eq = self.enhance_contrast(np.asarray(img_temp))
        temp_eq = temp_eq.astype(np.uint8)
        
        if size == 3:
            img_eq_hsv = np.dstack(tup = (img_hsv[:,:,0], img_hsv[:,:,1], temp_eq))
            img_eq = cv2.cvtColor(img_eq_hsv, cv2.COLOR_HSV2RGB)

            
        elif size == 2:
            img_eq = temp_eq

        
        self.display_image(img_eq)
        print('histogram equilisation done')
        
    
    
    def gamma_correction(self):
        
        alpha = (self.gamma_slider.value())/10          #assign alpha from front-end slider
        print(alpha)
        
        img = self.image
        
        size = len(np.shape(img))
        if size == 3:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_temp = img_hsv[:,:,2]
            print
    
        elif size == 2:
            img_temp = img
            
        img_temp_gamma = ((img_temp/255)**(alpha))*255
        img_temp_gamma = img_temp_gamma.astype(np.uint8)
        
        if size == 3:
            img_eq_hsv = np.dstack(tup = (img_hsv[:,:,0], img_hsv[:,:,1], img_temp_gamma))
            img_gamma = cv2.cvtColor(img_eq_hsv, cv2.COLOR_HSV2RGB)

            
        elif size == 2:
            img_gamma = img_temp_gamma

        
        self.display_image(img_gamma)
        print('gamma correction done')

        
    
    def log_transformation(self):
        print('log transformation done')
        pass
    
    def sharpen(self):
        val = self.sharpen_slider.value()
        print(val)
        print('image sharpening done')
        
        pass
    
    def enhance_contrast(self,image_matrix, bins=256):
        #img_ary = np.asarray(image_matrix)
        image_flattened = image_matrix.flatten()
        image_hist = np.zeros(bins)
    
        # frequency count of each pixel
        for pix in image_matrix:
            image_hist[pix] += 1
    
        # cummulative sum
        cum_sum = np.cumsum(image_hist)
        norm = (cum_sum - cum_sum.min()) * 256
        # normalization of the pixel values
        n_ = cum_sum.max() - cum_sum.min()
        uniform_norm = norm / n_
        uniform_norm = uniform_norm.astype('int')
    
        # flat histogram
        image_eq = uniform_norm[image_flattened]
        # reshaping the flattened matrix to its original shape
        image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

        return image_eq
    
    def save_image(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('ImaqArt'+str(self.n)+'.png',img)
        self.n += 1
        print ('image saved: ' + str(self.n))
    
    def undo_all(self):
        self.display_image(self.org_image)
        
    def undo(self):
        self.display_image(self.last_image)
        
        
        
    
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QMainWindow()
    prog = ImaqArt(main)
    main.show()
    sys.exit(app.exec_())