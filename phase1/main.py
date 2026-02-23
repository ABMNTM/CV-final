import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
import os
from skimage.measure import shannon_entropy
from scipy.stats import skew

from sklearn.svm import SVC
from pickle import load as model_load, dump as model_dump


class ImageQualityAanlyzer:
    IMAGE_SIZE = 350

    def __init__(self, assets_path: str):
        self.assets_path = Path(assets_path).resolve()

    def create_dataset(self):
        train_norm_dir = self.assets_path / "Real_captured/Train/Normal"
        train_low_dir = self.assets_path / "Real_captured/Train/Low"
        test_norm_dir = self.assets_path / "Real_captured/Test/Normal"
        test_low_dir = self.assets_path / "Real_captured/Test/Low"
        normal_images = [train_norm_dir / item for item in os.listdir(train_norm_dir)]
        low_images = [train_low_dir / item for item in os.listdir(train_low_dir)]
        whole_size = len(normal_images) + len(low_images)
        self.train_dataframe = pd.DataFrame(
            np.nan,
            index=range(whole_size),
            columns=(
                "Mean intensity",
                "Standard Deviation",
                "Entropy",
                "Histogram Skewness",
            )
        )
        self.train_class = pd.Series(index=range(whole_size), data=np.nan)
        i = 0
        self.fill_dataset_for_images(self.train_dataframe, normal_images, i)
        i = len(normal_images)
        self.train_class[0:i] = 1 # normal
        self.train_class[i:whole_size] = 0 # low
        self.fill_dataset_for_images(self.train_dataframe, low_images, i)
        
        self.train_dataframe.to_csv("train_dataset.csv")
        self.train_class.to_csv("train_classes.csv")

        normal_images = [test_norm_dir / item for item in os.listdir(test_norm_dir)]
        low_images = [test_low_dir / item for item in os.listdir(test_low_dir)]
        whole_size = len(normal_images) + len(low_images)
        self.test_dataframe = pd.DataFrame(
            np.nan,
            index=range(whole_size),
            columns=(
                "Mean intensity",
                "Standard Deviation",
                "Entropy",
                "Histogram Skewness",
            )
        )
        self.test_class = pd.Series(index=range(whole_size), data=np.nan)
        i = 0
        self.test_class
        self.fill_dataset_for_images(self.test_dataframe, normal_images, i)
        i = len(normal_images)
        self.test_class[0:i] = 1 # normal
        self.test_class[i:whole_size] = 0 # low
        self.fill_dataset_for_images(self.test_dataframe, low_images, i)
        
        self.test_dataframe.to_csv("test_dataset.csv")
        self.test_class.to_csv("test_classes.csv")

    def fill_dataset_for_images(self, df, img_list, i):
        for image_path in img_list:
            image = cv.imread(image_path)
            if image is None:
                raise FileNotFoundError(image_path)
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # preprocessing
            image_float_resized = cv.resize(image_gray, (self.IMAGE_SIZE, self.IMAGE_SIZE)).astype(np.float32)
            final_image = cv.normalize(image_float_resized, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
            df.iloc[i, 0] = np.mean(final_image, dtype=np.float32)
            df.iloc[i, 1] = np.std(final_image, dtype=np.float32)
            df.iloc[i, 2] = shannon_entropy(final_image)
            df.iloc[i, 3] = skew(final_image.flatten())
            i += 1

    def learn_model(self):
        self.model = SVC(kernel="rbf", coef0=0)
        self.model.fit(self.train_dataframe, self.train_class)
        with open("model.pkl", "wb") as file:
            model_dump(self.model, file)
        self.model

    def report_model(self):
        from sklearn.metrics import classification_report, confusion_matrix
        y_predict = self.model.predict(self.test_dataframe)
        report = classification_report(self.test_class, y_predict)
        confusion_mat = confusion_matrix(self.test_class, y_predict)
        with open("report.txt", "wt") as file:
            file.write("classification reports\n\n\n")
            file.write(report)
            file.write("\n\n")
            file.write(str(confusion_mat))


if __name__ == "__main__":
    im = ImageQualityAanlyzer("assets")
    print("phase 4.1 started.")
    im.create_dataset()
    print("dataset created successfully")
    im.learn_model()
    print("model performed.")
    im.report_model()
    print("operation finished.")
