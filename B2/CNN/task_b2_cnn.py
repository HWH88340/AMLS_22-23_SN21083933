import tensorflow as tf
import glob
import pandas as pd
import os
import numpy as np
import cv2

def mkdir(path):
    directory = os.path.exists(path)
    if not directory:
        os.makedirs(path)

def img_convert(inputfile, outputdir): # Haar Cascade
    img = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)

    try:
        face_cascade = cv2.CascadeClassifier(
            '/Users/33381/anaconda3/envs/ELEC0134_py36_cv/Lib/site-packages/cv2/data/haarcascade_eye.xml')  # eyes

        # detects faces in the input image
        face_detect = face_cascade.detectMultiScale(img, 1.3, 4)
        print('Number of detected faces:', len(face_detect))

        # loop over all detected faces

        if len(face_detect) > 0:
            for i, (x, y, w, h) in enumerate(face_detect):
                face_img = img[y :y + h, x:x + w]
                face_img_2 = cv2.resize(face_img, (32, 32), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(outputdir, os.path.basename(inputfile)), face_img_2)
    except Exception as e:
        print(e)


def read_picture(train_flag):
    #preprocessing
    if train_flag == 1:

        j = 0
        for inputfile in glob.glob("./Datasets/cartoon_set/img/*.png"):
            outputdir = "./b2_cnn_processed_eyes"
            mkdir(outputdir)
            img_convert(inputfile, outputdir)
            j = j + 1
            print(j, "times")

        # filequeue
        file_name_list = glob.glob("./b2_cnn_processed_eyes/*.png")
    else:
        j = 0
        for inputfile in glob.glob("./Datasets/cartoon_set_test/img/*.png"):
            outputdir = "./b2_cnn_processed_eyes_test"
            mkdir(outputdir)
            img_convert(inputfile, outputdir)
            j = j + 1
            print(j, "times")

        # filequeue
        file_name_list = glob.glob("./b2_cnn_processed_eyes_test/*.png")


    file_name_queue = tf.train.string_input_producer(file_name_list)

    # decode
    reader = tf.WholeFileReader()
    file, img = reader.read(file_name_queue)    # width = 500, height = 500, channel = 4
    image_dec = tf.image.decode_png(img)
    image_dec.set_shape([32, 32, 1])
    image_new = tf.cast(image_dec, tf.float32)

    if train_flag == 1:
        filename_batch, image_batch = tf.train.batch([file, image_new], batch_size=10, num_threads=2, capacity=100)
    else:
        filename_batch, image_batch = tf.train.batch([file, image_new], batch_size=1)

    return filename_batch, image_batch


# connect filename with label
def filename2label(files, csv_data):
    label_queue = []

    for file in files:
        digit_str_name = "".join(list(filter(str.isdigit, str(file))))
        label = csv_data.loc[int(digit_str_name), "eye_color"]
        label_queue.append(label)


    return np.array(label_queue)


def create_model(x):
    # 32, 32, 1
    # 1???Conv1
    # weights, bias, conv, relu1, and pool for conv1
    weights_conv1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32], stddev=0.01))
    bias_conv1 = tf.Variable(initial_value=tf.random_normal(shape=[32], stddev=0.01))
    x_conv1 = tf.nn.conv2d(input=x, filter=weights_conv1, strides=[1, 1, 1, 1], padding="SAME") + bias_conv1
    x_relu1 = tf.nn.relu(x_conv1)
    x_pool1 = tf.nn.max_pool(value=x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2???Conv2
    # 32, 32, 1 -> 16, 16, 64
    # weights, bias, conv, relu1, and pool for conv2
    weights_conv2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64], stddev=0.01))
    bias_conv2 = tf.Variable(initial_value=tf.random_normal(shape=[64], stddev=0.01))
    x_conv2 = tf.nn.conv2d(input=x_pool1, filter=weights_conv2, strides=[1, 1, 1, 1], padding="SAME") + bias_conv2
    x_relu2 = tf.nn.relu(x_conv2)
    x_pool2 = tf.nn.max_pool(value=x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3???Full connection
    # [None, 16, 16, 32] -> [None, 8, 8, 64]
    # [None, 8, 8, 64] -> [None, 8 * 8 * 64]
    # [None, 8 * 8 * 64] * [8 * 8 * 64, 5] = [None, 5]
    x_fc = tf.reshape(x_pool2, shape=[-1, 8 * 8 * 64])
    weights_fc = tf.Variable(initial_value=tf.random_normal(shape=[8 * 8 * 64, 5], stddev=0.01))
    bias_fc = tf.Variable(initial_value=tf.random_normal(shape=[5], stddev=0.01))
    y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def task_b2_cnn(train_flag):
    filename, image = read_picture(train_flag)
    if train_flag == 1:
        data_csv = pd.read_csv("./Datasets/cartoon_set/labels.csv", delimiter='\t', index_col=0)
    else:
        data_csv = pd.read_csv("./Datasets/cartoon_set_test/labels.csv", delimiter='\t', index_col=0)
    # prepare data
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    y_true = tf.placeholder(tf.float32, shape=[None, 5])

    # construct models
    y_predict = create_model(x)

    # construct loss function
    list_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    losses = tf.reduce_mean(list_loss)

    # optimize the loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(losses)

    # calculate the loss
    list_equal = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracies = tf.reduce_mean(tf.cast(list_equal, tf.float32))

    # record
    saver = tf.train.Saver()
    initial = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:

        # initializaiton
        sess.run(initial)

        if train_flag == 1:
            # start the coord and threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(2000):
                filename_value, image_value = sess.run([filename, image])

                labels = filename2label(filename_value, data_csv)
                # one-hot
                labels_value = tf.reshape(tf.one_hot(labels, depth=5), [-1, 5]).eval()

                _, error, accuracy_value = sess.run([optimizer, losses, accuracies],
                                                    feed_dict={x: image_value, y_true: labels_value})

                print("Training times: %d, Loss: %f???Accuracy: %f" % (i + 1, error, accuracy_value))

                if i % 30 == 0:
                    saver.save(sess, "./model_B2/Task_B2_model")
                if error < 0.001:
                    break

            coord.request_stop()
            coord.join(threads)
        else:
            # load the model
            if os.path.exists("./model_B2/checkpoint"):
                saver.restore(sess, "./model_B2/Task_B2_model")
            # start the coord
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            correct = 0
            # prediction
            for i in range(1000):
                # predict with one sample once
                filename_value, image_value = sess.run([filename, image])
                labels = filename2label(filename_value, data_csv)
                # one-hot encoding
                labels_value = tf.reshape(tf.one_hot(labels, depth=5), [-1, 5]).eval()

                true = tf.argmax(sess.run(y_true, feed_dict={x: image_value, y_true: labels_value}), 1).eval()
                predict = tf.argmax(sess.run(y_predict, feed_dict={x: image_value, y_true: labels_value}), 1).eval()
                if true == predict:
                    correct = correct + 1
                accuracy = correct / (i + 1)
                print("Time: %d, Y true: %d, Y prediction: %d, accuracy: %f" % (i + 1, true, predict, accuracy)
                      )
            coord.request_stop()
            coord.join(threads)






