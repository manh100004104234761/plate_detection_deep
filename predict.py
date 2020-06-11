import get_chars
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt
from sklearn import metrics


# Load model
model = load_model('cnn_classifier.h5')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Path to the plate image")
ap.add_argument("-f", "--folder", default=None,
                help="Path to the folder plate images")
args = vars(ap.parse_args())
find_char = False
get_accuracy = False
if args['image'] is not None:
    find_char = True
    if not os.path.exists(args['image']):
        print("The image path doesn't exist!")
        find_char = False
if args['folder'] is not None:
    get_accuracy = True
    if not os.path.exists(args['folder']):
        print("The folder path doesn't exist!")
        get_accuracy = False
args = vars(ap.parse_args())


def recognize_char(list_of_chars):
    str_plate = ""
    for char in list_of_chars:
        char = np.reshape(char, (1, 28, 28, 1))
        out = model.predict(char)
        p = []
        precision = 0
        for i in range(len(out)):
            z = np.zeros(17)
            z[np.argmax(out[i])] = 1.
            precision = max(out[i])
            p.append(z)
        prediction = np.array(p)
        # Inverse one hot encoding
        alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                     'B', 'C', 'D', 'E', 'G', 'H']
        classes = []
        for a in alphabets:
            classes.append([a])
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(classes)
        pred = ohe.inverse_transform(prediction)

        if precision > 0.1:
            str_plate += str(pred[0][0])
    return str_plate


def main():
    if find_char:
        img = cv2.imread(args['image'])
        img_gray, img_thresh = get_chars.preprocess(img)
        list_of_chars = get_chars.find_chars(img_thresh)
        str_plate = recognize_char(list_of_chars)
        print(str_plate)
    elif get_accuracy:
        count_images = 0
        count_true_plate = 0
        count_chars = 0
        count_true = 0
        str_plates = []
        labels = []
        for image in os.listdir(args['folder']):
            count_images += 1
            label = image[9:17]
            labels += label
            pathImage = args['folder'] + "/" + image
            img = cv2.imread(pathImage)
            img_gray, img_thresh = get_chars.preprocess(img)
            list_of_chars = get_chars.find_chars(img_thresh)
            str_plate = recognize_char(list_of_chars)
            if len(str_plate) < len(label):
                for i in range(0, len(label) - len(str_plate)):
                    str_plate = str_plate + '0'
            str_plates += str_plate
            for i in range(0, len(label)):
                count_chars += 1
                if label[i] == str_plate[i]:
                    count_true += 1
            if (label == str_plate):
                count_true_plate += 1
            else:
                print(label)
                print(str_plate + "\n")
        confusion_matrix = metrics.confusion_matrix(labels, str_plates)
        print("Total chars: " + str(count_chars) + "\n")
        print("True chars: " + str(count_true) + "\n")
        print("Accuracy chars: " + str(float(count_true/count_chars)) + "\n")
        print("Accuracy plate: " + str(float(count_true_plate/count_images)))
        print("confusion matrix: \n{}".format(confusion_matrix))
        report = metrics.classification_report(labels, str_plates)
        print(report)
        return
    else:
        print("Nothing to do, pass args to do some thing!")
    return


if __name__ == '__main__':
    main()
