import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
import random
import os
from skimage.measure import shannon_entropy
from scipy.stats import skew


class ImageQualityAanlyzer:
    IMAGE_SIZE = 350

    def __init__(self, assets_path: str):
        self.assets_path = Path(assets_path).resolve()

    def create_dataset(self):
        normal_images = os.listdir(self.assets_path / "Real_captured/Train/Normal")
        low_images = os.listdir(self.assets_path / "Real_captured/Train/Low")
        self.train_dataframe = pd.DataFrame(
            np.nan,
            index=len(normal_images) + len(low_images),
            columns=(
                "ID",
                "Mean intensity",
                "Standard Deviation",
                "Entropy",
                "Histogram Skewness",
                "class",
            )
        )
        i = 0
        self.fill_dataset_for_images(normal_images, i, 1)
        i = len(normal_images)
        self.fill_dataset_for_images(low_images, i, 0)
        
        self.train_dataframe.to_csv("train_dataset.csv")
        
        normal_images = os.listdir(self.assets_path / "Real_captured/Test/Normal")
        low_images = os.listdir(self.assets_path / "Real_captured/Test/Low")
        self.test_dataframe = pd.DataFrame(
            np.nan,
            index=len(normal_images) + len(low_images),
            columns=(
                "ID",
                "Mean intensity",
                "Standard Deviation",
                "Entropy",
                "Histogram Skewness",
                "class",
            )
        )
        i = 0
        self.fill_dataset_for_images(normal_images, i, 1)
        i = len(normal_images)
        self.fill_dataset_for_images(low_images, i, 0)
        
        self.train_dataframe.to_csv("test_dataset.csv")

            
    def fill_dataset_for_images(self, img_list, i, class_of_lst):
        for image_path in img_list:
            image = cv.imread(image_path)
            if not image:
                raise FileNotFoundError
            image_gray = cv.cvtColor(cv.COLOR_BGR2GRAY)

            # preprocessing
            image_resized = cv.resize(image_gray, self.IMAGE_SIZE)
            final_image = cv.normalize(
                image_resized, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX
            )
            reformed_image = final_image.astype(np.float32)

            self.train_dataframe.iloc[i, "ID"] = image_path
            self.train_dataframe.iloc[i, "Mean intensity"] = np.mean(reformed_image)
            self.train_dataframe.iloc[i, "Standard Deviation"] = np.std(reformed_image)
            self.train_dataframe.iloc[i, "Entropy"] = shannon_entropy(reformed_image)
            self.train_dataframe.iloc[i, "Histogram Skewness"] = skew(reformed_image.flatten())
            self.train_dataframe.iloc[i, "class"] = class_of_lst
            i += 1

            # cv.imshow("", final_image)
            # cv.waitKey(0.1)
            
    # def learn_model(self):
    #     model = 


im = ImageQualityAanlyzer("../assets")
im.create_dataset()