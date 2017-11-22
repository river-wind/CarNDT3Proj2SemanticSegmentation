import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


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
    # Implement load_vgg function
    #following walkthrough process, use model loader to load layers
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # load tensors
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

#run basic tests
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
    # Implement layers function
    # 1x1 convolution of vgg_layer7
    conv71x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), 
                                padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  # penalize if the weights get too large

    # upsample by 2
    upsamp7_2x = tf.layers.conv2d_transpose(conv71x1, num_classes, 4, 
                                strides=(2, 2), 
                                padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))   # penalize if the weights get too large
    tf.Print(upsamp7_2x, [tf.shape(upsamp7_2x)[1:3]])
    #1x1 convolution of vgg_layer4
    conv41x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), 
                                padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))   # penalize if the weights get too large

    # add skip connection
    skip_add_1 = tf.add(upsamp7_2x, conv41x1)

    #upsample the skip layer by 2
    upsampskip1_2x = tf.layers.conv2d_transpose(skip_add_1, num_classes, 4, 
                                strides=(2, 2), 
                                padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))   # penalize if the weights get too large

    tf.Print(upsampskip1_2x, [tf.shape(upsampskip1_2x)[1:3]]) 
    #1x1 convolution of vgg_layer3
    conv31x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), 
                                padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))   # penalize if the weights get too large


    # add skip connection and upsample by 8
    skip_add_2 = tf.add(upsampskip1_2x, conv31x1)

    output = tf.layers.conv2d_transpose(skip_add_2, num_classes, 16, 
                                strides=(8, 8), 
                                padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))   # penalize if the weights get too large

    return output

#test layers
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
    
    # reshape parameters to 2D tensors
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    #loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    #set Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    # Use the optimizer to minimize loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

#run tests
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
    cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
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
    sess.run(tf.global_variables_initializer())
    # 4: Implement function
    print("Training...")

    for epoch in range(epochs):
      for z, (image, label) in enumerate(get_batches_fn(batch_size)):
        # Training
        _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:image, correct_label:label, keep_prob:0.4, keep_prob: 0.5, learning_rate: 0.001}) 

        print("epoch: {}  batch: {}  loss: {}".format(epoch+1, z, loss))
      
#run tests
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
        # Get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 40
        batches = 5

        # Build the network
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        label = tf.placeholder(tf.int32, [None, None, None, num_classes]) 
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, loss = optimize(final_layer, label, learning_rate, num_classes)

        #Save the trained model if we want to re-use it after training
        #saver = tf.train.Saver()
        #saver.save(sess, './runs/sem_seg_model.ckpt')
        #restore a saved model:
        #saver.restore(sess, './runs/sem_seg_model.ckpt')

        #Train the network
        train_nn(sess, epochs, batches, get_batches_fn, train_op, loss, input_image, label, keep_prob, learning_rate)

        # Save the inference data to output images
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()
