import tensorflow as tf
import glob
import pandas as pd
import os
import numpy as np

tf.app.flags.DEFINE_integer("is_train", 1, "1 for training, 0 for predicting")
FLAGS = tf.app.flags.FLAGS


def read_picture(train):
    # filequeue
    if train == 1:
        file_name_list = glob.glob("./img/*.png")
    else:
        file_name_list = glob.glob("./img/test/*.png")
    file_name_queue = tf.train.string_input_producer(file_name_list)

    # decode
    reader = tf.WholeFileReader()
    file, img = reader.read(file_name_queue)    # width = 500, height = 500, channel = 4
    image_dec = tf.image.decode_png(img)
    image_dec.set_shape([500, 500, 4])
    image_new = tf.cast(image_dec, tf.float32)

    if train == 1:
        filename_batch, image_batch = tf.train.batch([file, image_new], batch_size=10, num_threads=2, capacity=100)
    else:
        filename_batch, image_batch = tf.train.batch([file, image_new], batch_size=1)

    return filename_batch, image_batch


# connect filename with label
def filename2label(files, csv_data):
    label_queue = []

    for file in files:
        digit_str_name = "".join(list(filter(str.isdigit, str(file))))
        label = csv_data.loc[int(digit_str_name), "face_shape"]
        # label = csv_data.loc[int(digit_str_name), "eye_color"]
        label_queue.append(label)


    return np.array(label_queue)


def create_model(x):
    # 500, 500, 4
    # 1）Conv1
    # weights, bias, conv, relu1, and pool for conv1
    weights_conv1 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 4, 32], stddev=0.01))
    bias_conv1 = tf.Variable(initial_value=tf.random_normal(shape=[32], stddev=0.01))
    x_conv1 = tf.nn.conv2d(input=x, filter=weights_conv1, strides=[1, 1, 1, 1], padding="SAME") + bias_conv1
    x_relu1 = tf.nn.relu(x_conv1)
    x_pool1 = tf.nn.max_pool(value=x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2）Conv2
    # 500, 500, 4 -> 250, 250, 32
    # weights, bias, conv, relu1, and pool for conv2
    weights_conv2 = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64], stddev=0.01))
    bias_conv2 = tf.Variable(initial_value=tf.random_normal(shape=[64], stddev=0.01))
    x_conv2 = tf.nn.conv2d(input=x_pool1, filter=weights_conv2, strides=[1, 1, 1, 1], padding="SAME") + bias_conv2
    x_relu2 = tf.nn.relu(x_conv2)
    x_pool2 = tf.nn.max_pool(value=x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3）Full connection
    # [None, 250, 250, 32] -> [None, 125, 125, 64]
    # [None, 125, 125, 64] -> [None, 125 * 125 * 64]
    # [None, 125 * 125 * 64] * [125 * 125 * 64, 5] = [None, 5]
    x_fc = tf.reshape(x_pool2, shape=[-1, 125 * 125 * 64])
    weights_fc = tf.Variable(initial_value=tf.random_normal(shape=[125 * 125 * 64, 5], stddev=0.01))
    bias_fc = tf.Variable(initial_value=tf.random_normal(shape=[5], stddev=0.01))
    y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def task_b1_cnn():
    filename, image = read_picture(1)
    data_csv = pd.read_csv("./labels.csv", delimiter='\t', index_col=0)

    # prepare data
    x = tf.placeholder(tf.float32, shape=[None, 500, 500, 4])
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

        # load the model
        if os.path.exists("./model_B1/checkpoint"):
            saver.restore(sess, "./model_B1/Task_B1_model")

        if FLAGS.is_train == 1:
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

                print("Training times: %d, Loss: %f，Accuracy: %f" % (i + 1, error, accuracy_value))

                if i % 30 == 0:
                    saver.save(sess, "./model_B1/Task_B1_model")
                if error < 0.001:
                    break

            coord.request_stop()
            coord.join(threads)
        else:
            # start the coord
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # prediction
            for i in range(100):
                # predict with one sample once
                filename, image = read_picture(0)
                filename_value, image_value = sess.run([filename, image])
                labels = filename2label(filename_value, data_csv)
                # one-hot encoding
                labels_value = tf.reshape(tf.one_hot(labels, depth=5), [-1, 5]).eval()

                print("Time: %d, Y true: %d, Y prediction: %d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: image_value, y_true: labels_value}), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: image_value, y_true: labels_value}), 1).eval()
                )
                      )

            coord.request_stop()
            coord.join(threads)





