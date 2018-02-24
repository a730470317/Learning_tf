import tensorflow as tf
from inspect import currentframe,getframeinfo

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

def get_filename():
    cf = currentframe()
    return  getframeinfo(cf).filename

print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data

MINIST_DATA_PATH = "/home/zivlin/Desktop/Learning_tf/minist_data/";
print(MINIST_DATA_PATH);
minist = input_data.read_data_sets(MINIST_DATA_PATH, one_hot=True);
print("Train:       ", minist.train.images.shape, minist.train.labels.shape);
print("Test:        ", minist.test.images.shape, minist.test.labels.shape);
print("Validation:  ", minist.validation.images.shape, minist.validation.labels.shape);
sess = tf.InteractiveSession()
# Define y = W*x+b
x = tf.placeholder(tf.float32, [None, 784]);
W = tf.Variable(tf.zeros([784, 10]));
b = tf.Variable(tf.zeros([10]));
y = tf.nn.softmax(tf.matmul(x, W) + b);
y_ = tf.placeholder(tf.float32, [None, 10]);
# Define cross entropy
print(get_filename(), get_linenumber())
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]));
print(get_filename(), get_linenumber())
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);
print(get_filename(), get_linenumber())
tf.global_variables_initializer().run();
print(get_filename(), get_linenumber())

sess = tf.Session()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32));

for i in range(1000):
    print(" i = ", i);
    batch_xs, batch_ys = minist.train.next_batch(1000);
    train_step.run({x: batch_xs, y_: batch_ys});
    print(accuracy.eval({x: minist.test.images, y_: minist.test.labels}));
print("===================");
print(accuracy.eval({x: minist.test.images, y_: minist.test.labels}));
print(minist)
