import tensorflow as tf

def conv_layer(inputs, shape, stride, freeze=None, padding='SAME', name=None):
	filter_size = tf.TensorShape(shape)
	f = tf.Variable(tf.truncated_normal(filter_size, stddev = 0.1), name = "filter")
	if freeze != None:
		f = tf.cond(freeze, lambda: tf.stop_gradient(f), lambda: f)
	conv = tf.nn.conv2d(inputs, f, stride, padding, name=name)
	return conv

class generator(object):
	def __init__(self, x):
		self.G = self._generate(x)
	
	def get_G(self):
		return self.G

	def _generate(self, imgs):
		'''
		imgs: batch_size*480*640*3
		return: batch_size*480*640*3
		'''
		#downsample
		stride = [1, 1, 1, 1]
		with tf.name_scope('G_conv1'):
			G_conv1 = conv_layer(imgs, [3, 3, 3, 64], stride, padding='SAME')
			G_conv1 = tf.nn.leaky_relu(G_conv1)
			G_pool1 = tf.nn.max_pool(G_conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv2'):
			G_conv2 = conv_layer(G_pool1, [3, 3, 64, 128], stride, padding='SAME')
			G_conv2 = tf.nn.leaky_relu(G_conv2)
			G_pool2 = tf.nn.max_pool(G_conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv3'):
			G_conv3 = conv_layer(G_pool2, [3, 3, 128, 256], stride, padding='SAME')
			G_conv3 = tf.nn.leaky_relu(G_conv3)
			G_pool3 = tf.nn.max_pool(G_conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

		with tf.name_scope('G_conv4'):
			G_conv4_1 = conv_layer(G_pool3, [3, 3, 256, 512], stride, padding='SAME')
			G_conv4_1 = tf.nn.leaky_relu(G_conv4_1)
			G_conv4_2 = conv_layer(G_conv4_1, [3, 3, 512, 512], stride, padding='SAME')
			G_conv4_2 = tf.nn.leaky_relu(G_conv4_2)

		#upsample
		with tf.name_scope('G_deconv3'):
			f_deconv3_shape = [3, 3, 256, 512]
			f_deconv3_shape = tf.TensorShape(f_deconv3_shape)
			f_deconv3 = tf.Variable(tf.truncated_normal(f_deconv3_shape, stddev=0.1))
			G_deconv3 = tf.nn.conv2d_transpose(G_conv4_2, f_deconv3, G_conv3.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv3 = tf.concat([G_deconv3, G_conv3], axis=3)
			G_deconv3 = conv_layer(G_deconv3, [3, 3, 512, 256], stride, padding='SAME')
			G_deconv3 = tf.nn.leaky_relu(G_deconv3)

		with tf.name_scope('G_deconv2'):
			f_deconv2_shape = [3, 3, 128, 256]
			f_deconv2_shape = tf.TensorShape(f_deconv2_shape)
			f_deconv2 = tf.Variable(tf.truncated_normal(f_deconv2_shape, stddev=0.1))
			G_deconv2 = tf.nn.conv2d_transpose(G_deconv3, f_deconv2, G_conv2.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv2 = tf.concat([G_deconv2, G_conv2], axis=3)
			G_deconv2 = conv_layer(G_deconv2, [3, 3, 256, 128], stride, padding='SAME')
			G_deconv2 = tf.nn.leaky_relu(G_deconv2)

		with tf.name_scope('G_deconv1'):
			f_deconv1_shape = [3, 3, 64, 128]
			f_deconv1_shape = tf.TensorShape(f_deconv1_shape)
			f_deconv1 = tf.Variable(tf.truncated_normal(f_deconv1_shape, stddev=0.1))
			G_deconv1 = tf.nn.conv2d_transpose(G_deconv2, f_deconv1, G_conv1.get_shape(), [1, 2, 2, 1], padding='SAME')
			G_deconv1 = tf.concat([G_deconv1, G_conv1], axis=3)
			G_deconv1 = conv_layer(G_deconv1, [3, 3, 128, 64], stride, padding='SAME')
			G_deconv1 = tf.nn.leaky_relu(G_deconv1)

		out = conv_layer(G_deconv1, [1, 1, 64, 3], stride, padding='SAME')
		return out

class discriminator(object):
	def __init__(self, x, freeze):
		self.D = self._discriminate(x, freeze)
	
	def get_D(self):
		return self.D

	def _discriminate(self, imgs, freeze):
		'''
		imgs: batch_size*480*640*6
		return: batch_size*2
		'''
		stride = [1, 1, 1, 1]
		keep_prob = 0.7
		conv = imgs
		channel_in = 6
		channel_out = 64
		for i in list(range(1,5)):
			with tf.name_scope('D_conv%d' % i):
				conv = conv_layer(conv, [3, 3, channel_in, channel_out], stride, freeze, padding='SAME', name='D_conv%d' % i)
				conv = tf.nn.leaky_relu(conv, name='D_leaky_relu%d' % i)
				conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='D_pool%d' % i)
				conv = tf.nn.dropout(conv, keep_prob)
				channel_in = channel_out
				channel_out *= 2

		dense = tf.reduce_mean(conv, axis=[1, 2])
		with tf.name_scope('D_dense'):
			wshape = [dense.get_shape()[-1], 2]
			wshape = tf.TensorShape(wshape)
			weight = tf.Variable(tf.truncated_normal(wshape, stddev=0.1))
			dense = tf.matmul(dense, weight)
			bias = tf.Variable(tf.truncated_normal([2], stddev=0.1))
			dense = tf.nn.bias_add(dense, bias)
		return dense
