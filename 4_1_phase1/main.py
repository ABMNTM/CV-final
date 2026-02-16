import numpy as np
import pandas as pd
import cv2 as cv
from pathlib import Path
import random
import os


class ImageQualityAanlyzer:
    TRAIN_SIZE = 100
    IMAGE_SIZE = 350

    def __init__(self, assets_path: str):
        self.assets_path = Path(assets_path).resolve()

    @staticmethod
    def move_random_item(source_list, destination_list):
        if not source_list:
            return None  # Handle empty list
        random_index = random.randrange(len(source_list))
        item = source_list.pop(random_index)
        destination_list.append(item)
        return item

    def create_dataset(self, dir_of_imgs: str):
        choosed_images = self.get_train_images(dir_of_imgs)
        dataframe = pd.DataFrame(
            np.nan,
            index=self.TRAIN_SIZE,
            columns=(
                "ID",
                "Mean intensity",
                "Standard Deviation",
                "Entropy",
                "Histogram Skewness",
            )
        )
        for image_path in choosed_images:
            image = cv.imread(image_path)
            if not image:
                raise FileNotFoundError
            image_gray = cv.cvtColor(cv.COLOR_BGR2GRAY)

            # preprocessing
            image_resized = cv.resize(image_gray, self.IMAGE_SIZE)
            final_image = cv.normalize(
                image_resized, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX
            )
            # TODO: calculate value of columns and full the dataframe.

    def get_train_images(self, dir_of_imgs: str) -> list[str]:
        try:
            if not os.path.isfile("choosen_images.txt"):
                raise FileNotFoundError
            with open("choosen_images.txt", "rt") as file:
                choosed_images = eval(file.read())  # returns a list
                assert type(choosed_images) == list, "invalid data"
        except (FileNotFoundError, AssertionError):
            choosed_images = []
            images_path = self.assets_path / dir_of_imgs
            remain_images = os.listdir(images_path)
            for _ in self.TRAIN_SIZE:
                self.move_random_item(remain_images, choosed_images)
                with open("choosen_images.txt", "wt") as file:
                    file.write(str(choosed_images))
        return choosed_images
