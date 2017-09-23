import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def main1():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x*x*y + y + 2
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        result = f.eval()
        print(result)


def main2():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)

def main3():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2/m*tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta-learning_rate*gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE = ", mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print(best_theta)

def main4():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE = ", mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print(best_theta)


def main5():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    n_epochs = 10
    learning_rate = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m/batch_size))

    X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        best_theta = theta.eval()
        print(best_theta)

def main6():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE = ", mse.eval())
                save_path = saver.save(sess, "/tmp/my_model.ckpt")
            sess.run(training_op)
        best_theta = theta.eval()
        print(best_theta)
        save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

def main7():


    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    n_epochs = 10
    learning_rate = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m/batch_size))

    X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        best_theta = theta.eval()
        print(best_theta)
    file_writer.close()


def main8():


    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    n_epochs = 10
    learning_rate = 0.01
    batch_size = 100
    n_batches = int(np.ceil(m/batch_size))

    X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        best_theta = theta.eval()
        print(best_theta)
    file_writer.flush()
    file_writer.close()


if __name__ == '__main__':
    #main1()
    #main2()
    #main3()
    #main4()
    #main5()
    #main7()
    main8()
