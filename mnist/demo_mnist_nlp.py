# 2018/2/25
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from inspect import currentframe, getframeinfo


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def get_filename():
    cf = currentframe()
    return getframeinfo(cf).filename

def get_mnist_data():
    mnist = input_data.read_data_sets(MINIST_DATA_PATH, one_hot=True);
    print("Train:       ", mnist.train.images.shape, mnist.train.labels.shape);
    print("Test:        ", mnist.test.images.shape, mnist.test.labels.shape);
    print("Validation:  ", mnist.validation.images.shape, mnist.validation.labels.shape);
    return mnist

print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data

MINIST_DATA_PATH = "./minist_data/";
print(MINIST_DATA_PATH);
print(get_filename(),get_linenumber())

sess = tf.InteractiveSession()
layer2_size = 784;
in_units = 784
h1_units = 300

# Define y = W*x+b
with tf.name_scope('Net'):
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1), name='W1');
    b1 = tf.Variable(tf.zeros([h1_units]), name="b1");
    W2 = tf.Variable(tf.zeros([h1_units, 10]), name="W2");
    b2 = tf.Variable(tf.zeros([10]), name="b2");
    x = tf.placeholder(tf.float32, [None, in_units], name='input_x');

    keep_prob = tf.placeholder(tf.float32);
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob);
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2);
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Define cross entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy');
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy);
print(get_filename(),get_linenumber())

tf.summary.FileWriter("logs_nlp/", sess.graph)

tf.global_variables_initializer().run();

sess = tf.Session()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
# Get Mnist data
mnist = get_mnist_data()
plot_y=[]
plot_x=[]
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(10000);
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75});
    plot_acc = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    plot_x.append(i)
    plot_y.append(plot_acc)
    print("Inter ", i, ", accuracy = ", plot_acc);
print("===================");

plt.plot(plot_x,plot_y)
plt.xlabel('train_step')
plt.ylabel('acc')
plt.show()

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}));
print(mnist)
