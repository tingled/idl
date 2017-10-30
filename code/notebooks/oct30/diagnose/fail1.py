import tensorflow as tf

from datasets import MNISTDataset
from time import time


# get the data
mnist = MNISTDataset("mnist_data", batch_size=256, seed=int(time()))


# define the model first, from input to output
imgs = tf.placeholder(tf.float32, shape=[None, 28*28])

n_hidden = 100
n_layers = 8
w_range = 0.15
hidden = imgs
for layer in range(n_layers):
    name = 'hidden_{}'.format(layer)
    hidden = tf.layers.dense(
            hidden, n_hidden, activation=tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(minval=-w_range, maxval=w_range),
            bias_initializer=tf.constant_initializer(0.01), name=name)
    with tf.name_scope('hidden'+str(layer)) as scope:
        tf.summary.histogram(name, hidden)

logits = tf.layers.dense(hidden, 10,
        kernel_initializer=tf.random_uniform_initializer(minval=-w_range, maxval=w_range),
        bias_initializer=tf.zeros_initializer, name="logits")


# create the cost and a single training step based on that
labels = tf.placeholder(tf.uint8, [None])
labels_onehot = tf.one_hot(indices=labels, depth=10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits))
tf.summary.scalar("cost", cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
for g, v in optimizer.compute_gradients(cross_entropy):
    tf.summary.scalar("grad_{}".format(v.name), tf.norm(g))
train_step = optimizer.minimize(cross_entropy)


# evaluation measures
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
merged = tf.summary.merge_all()

# train!
with tf.Session() as sess:
    writer = tf.summary.FileWriter('summaries/ex1/gv3', sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        batch_xs, batch_ys = mnist.next_batch()
        stats, _ = sess.run([merged, train_step], feed_dict={imgs: batch_xs, labels: batch_ys})
        writer.add_summary(stats, step)
        if step % 100 == 0:
            train_acc = sess.run([accuracy, cross_entropy], feed_dict={imgs: batch_xs, labels: batch_ys})
            print("Training accuracy on the {}th step: {}".format(step, train_acc))

    # now evaluate
    print("Final test accuracy: {}".format(sess.run([accuracy, cross_entropy],
                                                    feed_dict={imgs: mnist.test_data, labels: mnist.test_labels})))
