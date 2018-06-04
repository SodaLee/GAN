import tensorflow as tf
from utils import generator, discriminator
import cv2
import numpy as np

batch_size = 2
plain = np.zeros((210, 480, 640, 3), float)
real = np.zeros((210, 480, 640, 3), float)

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

def save_img(imgs, step):
	print("saving %d rounds img." % step)
	for i in range(batch_size):
		cv2.imwrite('output/train_' + str(step) + '/' + str(i).zfill(4) + '.png', imgs[i])


def train_net(maxiter=100, restore=False, check_path = '', K = 3):
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
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, check_path)
		else:
			sess.run(tf.global_variables_initializer())
			print('initial done')

		for step in range(maxiter):
			plain, real = readin()
			for iters in range(200 // batch_size):
				plain1 = plain[iters * batch_size: (iters + 1) * batch_size]
				real1 = real[iters * batch_size: (iters + 1) * batch_size]
				print(step, iters)
				feed_D = {ginputs: plain1,
						  dinputs: real1,
						  choice: False,
						  freeze_D: False}
				feed_G = {ginputs: plain1,
						  choice: True,
						  freeze_D: True}
				loss, _ = sess.run([loss, train_op], feed_dict = feed_D)
				print("step %d: %.3lf" % (step,loss))
				for i in range(K):
					loss, _ = sess.run([loss, train_op], feed_dict = feed_G)
					print("step %d: %.3lf" % (step,loss))
				feed_GD = {ginputs: plain1,
						  choice: True,
						  freeze_D: False}
				loss, _ = sess.run([loss, train_op], feed_dict = feed_GD)
				print("step %d: %.3lf" % (step,loss))

			if(step % log_n == 0):
				test_G = {ginputs: plain,
						  choice: True,
						  freeze_D: True}
				out_G = sess.run([G_out], feed_dict = test_G)
				save_img(out_G, step)
			saver.save(sess, check_path, global_step = step)

def predo(tot_num = 200):

	for i in range(tot_num):
		img = cv2.imread('plain/' + str(i).zfill(4) + '.png')
		# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		plain[i] = img
		img = cv2.imread('real/' + str(i).zfill(4) + '.png')
		# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		real[i] = img

def readin(tot_num = 200):
	index = np.arange(tot_num)
	np.random.shuffle(index)
	return plain[index], real[index]
	


if __name__ == '__main__':
	predo()
	train_net(check_path = 'model/')
