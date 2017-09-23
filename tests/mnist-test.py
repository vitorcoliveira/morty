import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# using learn api tensor-flow
def main1():
    mnist = input_data.read_data_sets("/tmp/data/")

    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    config = tf.contrib.learn.RunConfig(tf_random_seed=42) # not shown in the config

    feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=feature_cols, config=config)
    dnn_clf = tf.contrib.learn.SKCompat(dnn_clf) # if TensorFlow >= 1.1
    dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

    y_pred = dnn_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred['classes']))

    y_pred_proba = y_pred['probabilities']
    print(log_loss(y_test, y_pred_proba))


# plain tensorflow
def main2():
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z

    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_model_final.ckpt")

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")  # or better, use save_path
        X_new_scaled = mnist.test.images[:20]
        Z = logits.eval(feed_dict={X: X_new_scaled})
        y_pred = np.argmax(Z, axis=1)

    print("Predicted classes:", y_pred)
    print("Actual classes:   ", mnist.test.labels[:20])


def main3():
    mnist = input_data.read_data_sets("/tmp/data/")

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_model_final.ckpt")

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")  # or better, use save_path
        X_new_scaled = mnist.test.images[:20]
        Z = logits.eval(feed_dict={X: X_new_scaled})
        y_pred = np.argmax(Z, axis=1)

    print("Predicted classes:", y_pred)
    print("Actual classes:   ", mnist.test.labels[:20])

if __name__ == '__main__':
    #main1()
    #main2()
    main3()