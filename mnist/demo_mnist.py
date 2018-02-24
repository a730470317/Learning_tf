import tensorflow as tf
from inspect import currentframe, getframeinfo


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def get_filename():
    cf = currentframe()
    return getframeinfo(cf).filename


print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data

MINIST_DATA_PATH = "../minist_data/";
print(MINIST_DATA_PATH);
mnist = input_data.read_data_sets(MINIST_DATA_PATH, one_hot=True);
print("Train:       ", mnist.train.images.shape, mnist.train.labels.shape);
print("Test:        ", mnist.test.images.shape, mnist.test.labels.shape);
print("Validation:  ", mnist.validation.images.shape, mnist.validation.labels.shape);
sess = tf.InteractiveSession()
layer2_size = 784;
# Define y = W*x+b
with tf.name_scope('Net'):
    x = tf.placeholder(tf.float32, [None, 784], name='input_x');
    W = tf.Variable(tf.zeros([layer2_size, 10]), name='input_W');
    b = tf.Variable(tf.zeros([10]), name='B');
    # W2 = tf.Variable(tf.zeros([784, layer2_size]), name='input_W2');
    # b2 = tf.Variable(tf.zeros([layer2_size]), name='B2');
    # y1 = tf.matmul(x, W2) + b2;
    # y = tf.nn.softmax( tf.matmul(y1, W) + b , name="softmax_y" );
    y = tf.nn.softmax(tf.matmul(x, W) + b , name="softmax_y");
    y_ = tf.placeholder(tf.float32, [None, 10]);
    # Define cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='cross_entropy');

tf.summary.FileWriter("logs/", sess.graph)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);
tf.global_variables_initializer().run();

sess = tf.Session()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100);
    train_step.run({x: batch_xs, y_: batch_ys});
    print("Inter ", i, ", accuracy = ", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}));
print("===================");
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}));
print(mnist)
