import tensorflow as tf
from utils import generator, discriminator

batch_size = 64

def gans(ginputs, dinputs, choice1, choice2, freeze):
	gout = generator(ginputs).get_G()
	din = tf.cond(
		choice1,
		lambda: gout,
		lambda: tf.cond(
			choice2,
			lambda: dinputs,
			lambda: tf.concat([tf.stop_gradient(gout), dinputs], axis=3)))
	dout = discriminator(din, freeze).get_D()
	return gout, dout

def train_net(maxiter=10, restore=False):

	ginputs = tf.placeholder(tf.float32, [batch_size, 480, 640, 3])
	dinputs = tf.placeholder(tf.float32, [batch_size, 480, 640, 3])
	choice1 = tf.placeholder(tf.bool)
	choice2 = tf.placeholder(tf.bool)
	freeze = tf.placeholder(tf.bool)
	G_out, D_out = gans(ginputs, dinputs, choice1, choice2, freeze)


	smooth = 0.1
	real_label = tf.stop_gradient(tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], axis = 1))
	fake_label = tf.stop_gradient(tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis = 1))
	# D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_out_real, labels=real_label * (1-smooth)))
	# D_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_out_fake, labels=fake_label))
	# D_loss = tf.add(D_loss_real, D_loss_fake)
	# G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_out_fake, labels=fake_label * (1-smooth)))
	# D_train_op = tf.train.AdamOptimizer().minimize(D_loss)
	# G_train_op = tf.train.AdamOptimizer().minimize(G_loss)
	label = tf.cond(
		choice1,
		lambda: fake_label,
		lambda: tf.cond(
			choice2,
			lambda: real_label,
			lambda: tf.concat([fake_label, real_label], axis=1)))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_out, labels=label * (1-smooth)))
	train_op = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
	train_net()