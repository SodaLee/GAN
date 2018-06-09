import tensorflow as tf
import numpy as np

def conv_layer(inputs, shape, stride, padding='SAME', name=None):
	filter_size = tf.TensorShape(shape)
	f = tf.Variable(tf.truncated_normal(filter_size, stddev = 0.1), name = "filter")
	conv = tf.nn.conv2d(inputs, f, stride, padding, name=name)
	return conv, f

def build_generator(ginputs, is_train):
	stride = [1, 1, 1, 1]

	with tf.variable_scope('G'):
		with tf.name_scope('G_conv1'):
			G_conv1, _ = conv_layer(ginputs, [3, 3, 1, 16], stride, padding='SAME')
			G_conv1 = tf.layers.batch_normalization(inputs=G_conv1, trainable=True, training=is_train, name="bn1")
			G_conv1 = tf.nn.leaky_relu(G_conv1)
			G_pool1 = tf.nn.max_pool(G_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv2'):
			G_conv2, _ = conv_layer(G_pool1, [3, 3, 16, 32], stride, padding='SAME')
			G_conv2 = tf.layers.batch_normalization(inputs=G_conv2, trainable=True, training=is_train, name="bn2")
			G_conv2 = tf.nn.leaky_relu(G_conv2)
			G_pool2 = tf.nn.max_pool(G_conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv3'):
			G_conv3, _ = conv_layer(G_pool2, [3, 3, 32, 64], stride, padding='SAME')
			G_conv3 = tf.layers.batch_normalization(inputs=G_conv3, trainable=True, training=is_train, name="bn3")
			G_conv3 = tf.nn.leaky_relu(G_conv3)
			G_pool3 = tf.nn.max_pool(G_conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv4'):
			G_conv4_1, _ = conv_layer(G_pool3, [3, 3, 64, 128], stride, padding='SAME')
			G_conv4_1 = tf.layers.batch_normalization(inputs=G_conv4_1, trainable=True, training=is_train, name="bn4_1")
			G_conv4_1 = tf.nn.leaky_relu(G_conv4_1)

			G_conv4_2, _ = conv_layer(G_conv4_1, [3, 3, 128, 128], stride, padding='SAME')
			G_conv4_2 = tf.layers.batch_normalization(inputs=G_conv4_2, trainable=True, training=is_train, name="bn4_2")
			G_conv4_2 = tf.nn.leaky_relu(G_conv4_2)

		#upsample
		with tf.name_scope('G_deconv3'):
			f_deconv3_shape = [3, 3, 64, 128]
			f_deconv3_shape = tf.TensorShape(f_deconv3_shape)
			f_deconv3 = tf.Variable(tf.truncated_normal(f_deconv3_shape, stddev=0.1))
			G_deconv3 = tf.nn.conv2d_transpose(G_conv4_2, f_deconv3, G_conv3.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv3 = tf.concat([G_deconv3, G_conv3], axis=3)
			G_deconv3, _ = conv_layer(G_deconv3, [3, 3, 128, 64], stride, padding='SAME')
			G_deconv3 = tf.layers.batch_normalization(inputs=G_deconv3, trainable=True, training=is_train, name="bn1_d")
			G_deconv3 = tf.nn.leaky_relu(G_deconv3)

		with tf.name_scope('G_deconv2'):
			f_deconv2_shape = [3, 3, 32, 64]
			f_deconv2_shape = tf.TensorShape(f_deconv2_shape)
			f_deconv2 = tf.Variable(tf.truncated_normal(f_deconv2_shape, stddev=0.1))
			G_deconv2 = tf.nn.conv2d_transpose(G_deconv3, f_deconv2, G_conv2.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv2 = tf.concat([G_deconv2, G_conv2], axis=3)
			G_deconv2, _ = conv_layer(G_deconv2, [3, 3, 64, 32], stride, padding='SAME')
			G_deconv2 = tf.layers.batch_normalization(inputs=G_deconv2, trainable=True, training=is_train, name="bn2_d")
			G_deconv2 = tf.nn.leaky_relu(G_deconv2)

		with tf.name_scope('G_deconv1'):
			f_deconv1_shape = [3, 3, 16, 32]
			f_deconv1_shape = tf.TensorShape(f_deconv1_shape)
			f_deconv1 = tf.Variable(tf.truncated_normal(f_deconv1_shape, stddev=0.1))
			G_deconv1 = tf.nn.conv2d_transpose(G_deconv2, f_deconv1, G_conv1.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv1 = tf.concat([G_deconv1, G_conv1], axis=3)
			G_deconv1, f = conv_layer(G_deconv1, [3, 3, 32, 16], stride, padding='SAME')
			G_deconv1 = tf.layers.batch_normalization(inputs=G_deconv1, trainable=True, training=is_train, name="bn3_d")
			G_deconv1 = tf.nn.leaky_relu(G_deconv1)

		out, _ = conv_layer(G_deconv1, [1, 1, 16, 1], stride, padding='SAME')
		out = tf.nn.tanh(out)
		params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "G")
		return out, params

def build_discriminator(dinputs, gouts, keep_prob):
	d_params = []
	stride = [1, 1, 1, 1]
	conv = tf.concat([dinputs, gouts], 0)
	channel_in = 1
	channel_out = 16
	for i in list(range(1,4)):
		with tf.name_scope('D_conv%d' % i):
			conv, f = conv_layer(conv, [3, 3, channel_in, channel_out], stride, padding='SAME', name='D_conv%d' % i)
			conv = tf.nn.leaky_relu(conv, name='D_leaky_relu%d' % i)
			conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='D_pool%d' % i)
			conv = tf.nn.dropout(conv, keep_prob)
			d_params.append(f)
			channel_in = channel_out
			channel_out *= 2

	# dense = tf.reduce_mean(conv, axis=[1, 2])
	shape = conv.shape
	dense = tf.reshape(conv, [-1,shape[1]*shape[2]*shape[3]])
	with tf.name_scope('D_dense'):
		wshape = [dense.get_shape()[-1], 1]
		wshape = tf.TensorShape(wshape)
		weight = tf.Variable(tf.truncated_normal(wshape, stddev=0.1))
		dense = tf.matmul(dense, weight)
		d_params.append(weight)
		bias = tf.Variable(tf.truncated_normal([1], stddev=0.1))
		dense = tf.nn.bias_add(dense, bias)
		d_params.append(bias)
		# dense = tf.nn.sigmoid(dense)
	# dense = tf.clip_by_value(dense, 1e-7, 1e5)
	return dense, d_params

