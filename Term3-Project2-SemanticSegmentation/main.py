import os.path
import time
import tensorflow as tf
import helper
import warnings
from datetime import timedelta
from distutils.version import LooseVersion
import project_tests as tests

KEEP_PROB = 0.5
LR     = 0.00009
STD_DEV= 1e-3
L2_REG = 1e-3
EPOCHS = 30
BATCH_SIZE = 16  # was 5, reducing to see if improves

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # IMPLEMENTED?: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #BRENT: New code, as per walkthrough
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    graph = tf.get_default_graph()
    w1in = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1in, keep, w3out, w4out, w7out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # IMPLEMENTED: Implement function

    # Finalizing encoder
    # Converting output of VGG-16 to FCN layer via 1x1 convolution of vgg layer 7
    encoder_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    # Creating decoder
    # Upsample prev input to original image size
    upsample_pool4_in1 = tf.layers.conv2d_transpose(encoder_out, num_classes, 4,
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))

    # Create matching transposed 1x1 conv version of previous pooling layer (layer 4) for skip connection
    upsample_pool4_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    # Create skip connection (element-wise addition)
    upsample_pool4_out = tf.add(upsample_pool4_in1, upsample_pool4_in2)
    # Create another upsampled skip connection from pool 3
    upsample_pool3_in1 = tf.layers.conv2d_transpose(upsample_pool4_out, num_classes, 4,
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    # Turning vgg pool 3 layer into 1x1 convolution for use in skip connection
    upsample_pool3_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    # Create another skip connection
    upsample_pool3_out = tf.add(upsample_pool3_in1, upsample_pool3_in2)

    # Finalize all upsampling
    nn_last_layer = tf.layers.conv2d_transpose(upsample_pool3_out, num_classes, 16,
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(L2_REG))
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # IMPLEMENTED: Implement function
    # Create logits as a 2D tensor where each row represents a pixel, and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1,num_classes))

    # Loss function will be cross entropy
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    # Training op will use adam optimizer and will minimize cross entropy loss
    train_op = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cross_entropy_loss)

    #Return tuple
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #IMPLEMENTED: Implement function
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()
    for i in range(epochs):
        start_time = time.time()
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LR})
            print("Epoch: {}".format(i + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss), " Time: ", str(timedelta(seconds=(time.time() - start_time))))
        print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # IMPLEMENTED: Build NN using load_vgg, layers, and optimize function
        # Create TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        #Load pretrained vgg model and create modified network
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        #Optimize network
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        # IMPLEMENTED: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # IMPLEMENTED: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
