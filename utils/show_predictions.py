from matplotlib import pyplot as plt
import numpy as np


def show_predictions(images, predictions, ground_truths, file_to_save):
    number_of_images = len(images)

    fig = plt.figure(dpi=200, figsize=(len(images) * 5, 5))
    for index, (image, prediction, gt) in enumerate(zip(images, predictions, ground_truths)):
        ax = plt.subplot(1, number_of_images, index+1)
        ax.set_title("PRED: {} | GT: {}".format(prediction, gt))
        ax.imshow(images[index])

    plt.savefig(file_to_save)
    plt.close()


if __name__ == "__main__":
    import cv2
    image = cv2.imread("./N0311.jpg", cv2.IMREAD_COLOR)

    show_predictions([image] * 5, ["NONE"] * 5, ["AA"] * 5, "skuska.png")
