# import required libraries
import os.path
import glob
import cv2
import pandas as pd
import numpy as np
import sklearn.metrics
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

def mkdir(path):
    directory = os.path.exists(path)
    if not directory:
        os.makedirs(path)


def img_convert_edge(inputfile, outputdir):
    image = cv2.imread(inputfile)

    blur = cv2.GaussianBlur(image, (3, 3), 0)  # gauss
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # gradient
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)


    edge_detect = cv2.Canny(xgrad, ygrad, 50, 150)  # 50 low，150 high
    # edge_detect = cv.Canny(gray,50,150)   #
    edge_detect_2 = cv2.resize(edge_detect, (200, 200), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(outputdir, os.path.basename(inputfile)), edge_detect_2)


def img_convert(inputfile, outputdir): # Haar Cascade
    img = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)

    try:
        face_cascade = cv2.CascadeClassifier(
            '/Users/33381/anaconda3/envs/ELEC0134_py36_cv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') # face & smile

        # detects faces in the input image
        face_detect = face_cascade.detectMultiScale(img, 1.3, 4)
        print('Number of detected faces:', len(face_detect))

        # loop over all detected faces

        if len(face_detect) > 0:
            for i, (x, y, w, h) in enumerate(face_detect):
                # face_img = img[y + int(2*h/3):y + h, x:x + w]
                face_img = img[y :y + h, x:x + w]
                face_img_2 = cv2.resize(face_img, (30, 30), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(outputdir, os.path.basename(inputfile)), face_img_2)
    except Exception as e:
        print(e)



def task_b1_svm(train_flag):
    if train_flag == 1:
        j = 0
        for inputfile in glob.glob("./Datasets/cartoon_set/img/*.png"):
            outputdir = "./b1_svm_processed_edge"
            mkdir(outputdir)
            img_convert_edge(inputfile, outputdir)
            j = j + 1
            print(j, "times")

        path = "./b1_svm_processed_edge"

        datanames = os.listdir(path)
        csv_data = pd.read_csv("./Datasets/cartoon_set/labels.csv", delimiter='\t', index_col=0)
    else:
        j = 0
        for inputfile in glob.glob("./Datasets/cartoon_set_test/img/*.png"):
            outputdir = "./b1_svm_processed_edge_test"
            mkdir(outputdir)
            img_convert_edge(inputfile, outputdir)
            j = j + 1
            print(j, "times")

        path = "./b1_svm_processed_edge_test"

        datanames = os.listdir(path)
        csv_data = pd.read_csv("./Datasets/cartoon_set_test/labels.csv", delimiter='\t', index_col=0)


    image_list = []
    label_list = []


    # training data
    for data in datanames:
        image_path = ''.join([path, '/', data])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = [i for j in image for i in j]

        image_list.append(image)

        digit_str = "".join(list(filter(str.isdigit, str(data))))
        label = csv_data.loc[int(digit_str), "face_shape"]

        label_list.append(label) # for task b1, 2

    image_list = np.array(image_list)
    label_list = np.array(label_list)

    if train_flag == 1:
        train_X, test_X, train_y, test_y = train_test_split(image_list, label_list)
        model = SVC(kernel='linear').fit(train_X, train_y)
        with open('b1_svm.pickle', 'wb') as w:
            pickle.dump(model, w)

        score = model.score(test_X, test_y)
        print(score)

        test_predict_y = model.predict(test_X)
        train_predict_y = model.predict(train_X)

        print("train")
        print(sklearn.metrics.classification_report(train_y, train_predict_y))
        print("test")
        print(sklearn.metrics.classification_report(test_y, test_predict_y))

        print("matrix 1")
        print(confusion_matrix(train_y, train_predict_y))
        print("matrix 2")
        print(confusion_matrix(test_y, test_predict_y))
    else:
        with open('b1_svm.pickle', 'rb') as r:
            test_model = pickle.load(r)
        score_test = test_model.score(image_list, label_list)
        print(score_test)