import tensorflow as tf
from utils import generator, discriminator

batch_size = 64

def gans(ginputs, dinputs, choice, freeze_D):
	gout = generator(ginputs).get_G()
	gout = tf.cond(
		freeze_D,
		lambda: gout,
		lambda: tf.stop_gradient(gout))
	din = tf.cond(
		choice,
		lambda: tf.concat([gout, ginputs], axis = -1),
		lambda: tf.concat([dinputs, ginputs], axis = -1)
	)
	dout = discriminator(din, freeze_D).get_D()
	return gout, dout

def train_net(maxiter=10, restore=False):
	ginputs = tf.placeholder(tf.float32, [batch_size, 480, 640, 3])
	dinputs = tf.placeholder(tf.float32, [batch_size, 480, 640, 3])
	choice = tf.placeholder(tf.bool, [])
	freeze_D = tf.placeholder(tf.bool, [])
	G_out, D_out = gans(ginputs, dinputs, choice, freeze_D)

	smooth = 0.1
	real_label = tf.stop_gradient(tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], axis = 1))
	fake_label = tf.stop_gradient(tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis = 1))
	label = tf.cond(
		choice,
		lambda: fake_label,
		lambda: real_label)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_out, labels=label * (1-smooth)))
	train_op = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		if restore:
			pass
		else:
			sess.run(tf.global_variables_initializer())
			print('initial done')

if __name__ == '__main__':
	train_net()
