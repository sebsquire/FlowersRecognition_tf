'''
Multi class classification of flower types using CNN
Base model:
  - 5 convolutional layers w/ max pooling proceeding each of the first 4.
  - 1 fully connected layer
  - 1 output layer (softmax)
Model attains 71% accuracy in 30 epochs using:
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

# ---------- PARAMETERS ----------
IMG_SIZE = 128                          # resize image to this height and width
num_classes = 5                         # different flower types
epochs = 30                             # number of times model sees full data

conv_k_size = 3                         # kernel filter height/width
batch_size = 16                         # batch size
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
4321 images are now:
IMG_SIZE * IMG_SIZE * RGB attached to one hot class label flower type and ordered randomly
Data is comprised of a list containing: [0]: image data, [1]: class label
'''
# create image (arrays) and label (lists) for use in models
X_train = np.array([item[0] for item in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = [item[1] for item in train_data]
x_valid = np.array([item[0] for item in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_valid = [item[1] for item in test_data]

X_train = X_train / 255                 # normalising
x_valid = x_valid / 255                 # normalising

# ---------- MODELLING ----------
# Placeholder Variables
x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, 3])
y = tf.placeholder('float')

# to save well performing models
MODEL_NAME = 'flowers-{}-{}.model'.format(lr, '5conv-basic')

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

class ConvNNModel:
    def __init__(self):
        self.plh_x = x                                                                      # x placeholder
        self.plh_y = y                                                                      # y placeholder
        self.w_conv1 = tf.Variable(tf.random_normal([conv_k_size, conv_k_size, 3, 32]))     # model variables (weights)
        self.w_conv2 = tf.Variable(tf.random_normal([conv_k_size, conv_k_size, 32, 64]))    # model variables (weights)
        self.w_conv3 = tf.Variable(tf.random_normal([conv_k_size, conv_k_size, 64, 128]))   # model variables (weights)
        self.w_conv4 = tf.Variable(tf.random_normal([conv_k_size, conv_k_size, 128, 256]))  # model variables (weights)
        self.w_conv5 = tf.Variable(tf.random_normal([conv_k_size, conv_k_size, 256, 512]))  # model variables (weights)
        self.w_fc = tf.Variable(tf.random_normal([8 * 8 * 512, 1024]))                      # model variables (weights)
        self.w_out = tf.Variable(tf.random_normal([1024, num_classes]))                     # model variables (biases)
        self.b1 = tf.Variable(tf.random_normal([32]))                                       # model variables (biases)
        self.b2 = tf.Variable(tf.random_normal([64]))                                       # model variables (biases)
        self.b3 = tf.Variable(tf.random_normal([128]))                                      # model variables (biases)
        self.b4 = tf.Variable(tf.random_normal([256]))                                      # model variables (biases)
        self.b5 = tf.Variable(tf.random_normal([512]))                                      # model variables (biases)
        self.b_fc = tf.Variable(tf.random_normal([1024]))                                   # model variables (biases)
        self.b_out = tf.Variable(tf.random_normal([num_classes]))                           # model variables (biases)

    # helper function to simplify model function
    def conv2d(self, layer, W):
        return tf.nn.conv2d(layer, W, strides=[1, 1, 1, 1], padding='SAME')

    # helper function to simplify model function
    def maxpool2D(self, layer):
        # 2*2 pool (ksize), stride 2 so non-overlapping
        return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def model(self):
        # layer 1
        conv1 = self.conv2d(layer=x, W=self.w_conv1)
        conv1 = self.maxpool2D(conv1)
        # layer 2
        conv2 = self.conv2d(layer=conv1, W=self.w_conv2)
        conv2 = self.maxpool2D(conv2)
        # layer 3
        conv3 = self.conv2d(layer=conv2, W=self.w_conv3)
        conv3 = self.maxpool2D(conv3)
        # layer 4
        conv4 = self.conv2d(layer=conv3, W=self.w_conv4)
        conv4 = self.maxpool2D(conv4)
        # layer 5
        conv5 = self.conv2d(layer=conv4, W=self.w_conv5)
        # fully connected layer
        fc = tf.reshape(conv5, [-1, 8 * 8 * 512])
        fc = tf.matmul(fc, self.w_fc) + self.b_fc
        fc = tf.nn.relu(fc)
        fc = tf.nn.dropout(fc, keep_prob=dropout_keep_rate)

        output = tf.matmul(fc, self.w_out) + self.b_out
        return output

    def train(self, num_epochs, batch_size, train_imgs, train_lbls, test_imgs, test_lbls):
        prediction = self.model()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                epoch_loss = 0
                for _ in range(int(len(train_data) / batch_size)):
                    epoch_x, epoch_y = next_batch(batch_size, train_imgs, train_lbls)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch + 1, 'completed out of', num_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: test_imgs, y: test_lbls}))


if __name__ == "__main__":
    nn = ConvNNModel()
    nn.train(num_epochs=epochs,
             batch_size=batch_size,
             train_imgs=X_train,
             train_lbls=Y_train,
             test_imgs=x_valid,
             test_lbls=y_valid)

# un-comment below to save model
# model.save(MODEL_NAME)
