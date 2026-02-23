import cv2 as cv
import numpy as np
from scipy.stats import cumfreq


class ImageEnhancer:
    def __get_lab(self, img):
        lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        return cv.split(lab_img)
    
    def __get_bgr(self, l, a, b):
        enhanced_lab = cv.merge((l, a, b))
        return cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    def histogram_equalization(self, img):
        """ img is BGR """
        l, a, b = self.__get_lab(img)
        l_enhanced = cv.equalizeHist(l)
        return self.__get_bgr(l_enhanced, a, b)

    def CLAHE(self, img):
        """ img is BGR """
        l, a, b = self.__get_lab(img)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return self.__get_bgr(l_enhanced, a, b)

    def gamma(self, img):
        """ img is BGR """
        l, a, b = self.__get_lab(img)
