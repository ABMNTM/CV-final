import cv2 as cv
import numpy as np
from scipy.stats import cumfreq


class ImageEnhancer:
    def __get_hsv(self, img):
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        return cv.split(hsv_img)
    
    def __get_bgr(self, h, s, v):
        hsv_img = cv.merge((h, s, v))
        return cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)

    def histogram_equalization(self, img):
        """ img is BGR """
        h, s, v = self.__get_hsv(img)
        v_enhanced = cv.equalizeHist(v)
        return self.__get_bgr(h, s, v_enhanced)

    def CLAHE(self, img):
        """ img is BGR """
        h, s, v = self.__get_hsv(img)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        return self.__get_bgr(h, s, v_enhanced)

    def gamma(self, img):
        """ img is BGR """
        h, s, v = self.__get_hsv(img)
        gamma = 0.5
        alpha = 1
        v_enhanced = alpha * np.power(v / 255.0, gamma) * 255.0
        v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)
        return self.__get_bgr(h, s, v_enhanced)
