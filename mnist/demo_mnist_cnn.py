# 2018/2/25
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from utility import tool_debug as t_debug


def get_mnist_data(MINIST_DATA_PATH):
    print("Load data form : ", MINIST_DATA_PATH);
    mnist = input_data.read_data_sets(MINIST_DATA_PATH, one_hot=True);
    print("Train:       ", mnist.train.images.shape, mnist.train.labels.shape);
    print("Test:        ", mnist.test.images.shape, mnist.test.labels.shape);
    print("Validation:  ", mnist.validation.images.shape, mnist.validation.labels.shape);
    return mnist


def weight_variable(shape, _name=''):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=_name)


def bias_variable(shape, _name=''):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=_name)
    # return tf.Variable(initial)


def conv2d(x, W,name="conv2d"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)


def max_pool_2x2(x,name = "max_pool_2x2"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=name)


print("tensorflow version is", tf.__version__)
# print(t_debug.get_filename(), t_debug.get_linenumber());

MINIST_DATA_PATH = "./minist_data/";
sess = tf.InteractiveSession()

# Define y = W*x+b
with tf.name_scope('CNN_Net1'):
    x = tf.placeholder(tf.float32, [None, 784],name='x')
    y_ = tf.placeholder(tf.float32, [None, 10],name='y_')
    x_image = tf.reshape(x, [-1, 28, 28, 1],name='x_image');
    W_conv1 = weight_variable([5, 5, 1, 32],"W_conv1")
    b_conv1 = bias_variable([32],"b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print(t_debug.get_filename(), t_debug.get_linenumber());

with tf.name_scope('CNN_Net2'):
    W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
    b_conv2 = bias_variable([64],"b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,name='h_conv2')
    print(t_debug.get_filename(), t_debug.get_linenumber());
    h_pool2 = max_pool_2x2(h_conv2)
    print(t_debug.get_filename(), t_debug.get_linenumber());

    W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
    b_fc1 = bias_variable([1024],"b_fc1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64],"h_pool2_flat")
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='h_fc1')

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name="h_fc1_drop")
    W_fc2 = weight_variable([1024, 10],"W_fc2")
    b_fc2 = bias_variable([10],"b_fc2")
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='conv')
print(t_debug.get_filename(), t_debug.get_linenumber());

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]), name='cross_entropy');
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy);
print(t_debug.get_filename(), t_debug.get_linenumber());

# Define cross entropy
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy');
# train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy);
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
print(t_debug.get_filename(), t_debug.get_linenumber());

tf.summary.FileWriter("logs_cnn/", sess.graph)

tf.global_variables_initializer().run();

sess = tf.Session()

# Get Mnist data
mnist = get_mnist_data(MINIST_DATA_PATH)
plot_y = []
plot_x = []
for i in range(3000):
    batch = mnist.train.next_batch(1000);
    feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
    train_step.run(feed_dict)

    plot_acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    plot_x.append(i)
    plot_y.append(plot_acc)
    print("Inter ", i, ", accuracy = ", plot_acc);
print("===================");

plt.plot(plot_x, plot_y)
plt.xlabel('train_step')
plt.ylabel('acc')
plt.show()

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}));
print(mnist)
