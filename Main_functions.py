'''
Multi class classification of flower types using CNN
Base model:
  - 5 convolutional layers w/ max pooling proceeding each of the first 4.
  - 1 fully connected layer
  - 1 output layer (softmax)
Model attains 68% accuracy in 30 epochs using:
  - AdamOptimizer with lr=0.001, weight_decay=1e-6, keep_rate=0.95, batch_size=16

To run this you will need to change:
 - directory of image_folders_dir - point it to the folder containing all flower type folders

Python 3.6.7, Tensorflow 1.11.0
'''
import os
import numpy as np
import tensorflow as tf

# image_folders_dir is location of folders containing images of various flower types
image_folders_dir = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Flowers Recognition\\flowers'

IMG_SIZE = 128                          # resize image to this height and width
num_classes = 5                         # different flower types
epochs = 30                             # number of times model sees full data

batch_size = 16                         # batch_size
lr = 0.001                              # learning rate
dropout_keep_rate = 0.95                # dropout keep rate in fully connected layer

# Ask user to load or process - for first time need to process but subsequently can load data
# UNLESS IMG_SIZE is changed
print('Load pre-existing preprocessed data for training (L) or preprocess data (P)?')
decision1 = input()
if decision1 == 'P' or decision1 == 'p':
    from preprocessing import create_data
    train_data, test_data = create_data(image_folders_dir, IMG_SIZE)
elif decision1 == 'L' or decision1 == 'l':
    if os.path.exists('train_data.npy'):
        train_data = np.load('train_data.npy')
        test_data = np.load('test_data.npy')
    else:
        raise Exception('No preprocessed data exists in path, please preprocess some.')
else:
    raise Exception('Please retry and type L or P')

'''
Images are now:
IMG_SIZE * IMG_SIZE * RGB attached to one hot class label flower type and ordered randomly
Data is comprised of a list containing: [0]: image data, [1]: class label
'''
# derive image and label data from new data sets
train_data_imgs = [item[0] for item in train_data]
train_data_lbls = [item[1] for item in train_data]
test_data_imgs = [item[0] for item in test_data]
test_data_lbls = [item[1] for item in test_data]
# create arrays for us in models
X_train = np.array(train_data_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = train_data_lbls
x_valid = np.array(test_data_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_valid = test_data_lbls

X_train = X_train / 255                 # normalising
x_valid = x_valid / 255                 # normalising

# ---------- MODELLING ----------
# placeholder variables
x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, 3])
y = tf.placeholder('float')

# to save well performing models
MODEL_NAME = 'flowers-{}-{}.model'.format(lr, '5conv-basic')


# helper function to simplify model function
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    # helper function to simplify model function
def maxpool2D(x):
    # 2*2 pool (ksize), stride 2 so non overlapping
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_NN_model(x, num_classes, img_size, keep_rate):
    # 5 by 5 kernel, 3 input depth (RGB image), 32 output depth (convolutions)
    weights = { 'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
                'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
                'W_conv3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
                'W_conv4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
                'W_conv5': tf.Variable(tf.random_normal([3, 3, 256, 512])),
                'W_fc': tf.Variable(tf.random_normal([8 * 8 * 512, 1024])), # conv output layer size, num neurons in fc
                'out': tf.Variable(tf.random_normal([1024, num_classes]))}

    biases = { 'b_conv1': tf.Variable(tf.random_normal([32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_conv3': tf.Variable(tf.random_normal([128])),
               'b_conv4': tf.Variable(tf.random_normal([256])),
               'b_conv5': tf.Variable(tf.random_normal([512])),
               'b_fc': tf.Variable(tf.random_normal([1024])),  # conv output layer size, num neurons in fc
               'out': tf.Variable(tf.random_normal([num_classes]))}

    x = tf.reshape(x, shape=[-1, img_size, img_size, 3])
    # layer 1
    conv1 = conv2d(x, weights['W_conv1'])
    conv1 = maxpool2D(conv1)
    # layer 2
    conv2 = conv2d(conv1, weights['W_conv2'])
    conv2 = maxpool2D(conv2)
    # layer 3
    conv3 = conv2d(conv2, weights['W_conv3'])
    conv3 = maxpool2D(conv3)
    # layer 4
    conv4 = conv2d(conv3, weights['W_conv4'])
    conv4 = maxpool2D(conv4)
    # layer 5
    conv5 = conv2d(conv4, weights['W_conv5'])
    # fully connected
    fc = tf.reshape(conv5, [-1, 8 * 8 * 512])
    fc = tf.matmul(fc, weights['W_fc']) + biases['b_fc']
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc, keep_prob=keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


# helper function: create a batch of data for training from whole training set
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_neural_network(x, num_epochs, batch_size, train_imgs, train_lbls, test_imgs, test_lbls):
    prediction = conv_NN_model(x=x,
                               num_classes=num_classes,
                               img_size=IMG_SIZE,
                               keep_rate=dropout_keep_rate)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.99, beta2=0.9999).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_data)/batch_size)):
                epoch_x, epoch_y = next_batch(batch_size, train_imgs, train_lbls)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', num_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_imgs, y: test_lbls}))


train_neural_network(x=x,
                     num_epochs=epochs,
                     batch_size=batch_size,
                     train_imgs=train_data_imgs,
                     train_lbls=train_data_lbls,
                     test_imgs=test_data_imgs,
                     test_lbls=test_data_lbls)

# un-comment below to save model
# model.save(MODEL_NAME)
