import tensorflow as tf
from net import build_generator, build_discriminator
import cv2
import numpy as np

plain = np.zeros((210, 48, 64, 1), float)
real = np.zeros((210, 48, 64, 1), float)
batch_size = 10
maxiter = 200
log_n = 5
check_path = 'model/'

def predo(tot_num = 200):

	for i in range(tot_num):
		img = cv2.imread('plain/' + str(i).zfill(4) + '.png')
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (64, 48))
		plain[i] = np.reshape(img, (48, 64, 1))
		img = cv2.imread('real/' + str(i).zfill(4) + '.png')
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (64, 48))
		real[i] = np.reshape(img, (48, 64, 1))
	print("read done...")

def readin(tot_num = 200):

	index = np.arange(tot_num)
	np.random.shuffle(index)
	return plain[index], real[index]

def save_img(imgs, step):
	print("saving %d rounds img." % step)
	print(np.shape(imgs))
	for i in range(batch_size):
		cv2.imwrite('output/train_' + str(step) + '/' + str(i).zfill(4) + '.png', imgs[0][i])


def train(restore = False, K = 3):
	ginputs = tf.placeholder(tf.float32, [batch_size, 48, 64, 1])
	dinputs = tf.placeholder(tf.float32, [batch_size, 48, 64, 1])
	keep_prob = tf.placeholder(tf.float32)

	g_out, g_params = build_generator(ginputs)
	d_out, d_params = build_discriminator(dinputs, g_out, keep_prob)
	d_real = tf.slice(d_out, [0, 0], [batch_size, -1])
	d_fake = tf.slice(d_out, [batch_size, 0], [-1, -1])

	d_loss = -(tf.log(d_real) + tf.log(1 - d_fake))
	g_loss = -tf.log(d_fake)

	optimizer = tf.train.AdamOptimizer()

	d_train = optimizer.minimize(d_loss, var_list = d_params)
	g_train = optimizer.minimize(g_loss, var_list = g_params)

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
					plain1 = plain1 / 255
					plain1 = 2 * plain1 - 1
					real1 = real1 / 255
					real1 = 2 * real1 - 1
					print(step, iters)
					feed = {ginputs: plain1,
							dinputs: real1,
							keep_prob: 0.5}
					loss, _ = sess.run([d_loss, d_train], feed_dict = feed)
					print('D Loss: ', loss)
					for j in range(K):
						loss, _ = sess.run([g_loss, g_train], feed_dict = feed)
						print('G Loss:', loss)

				if(step % log_n == 0):
					test_G = {ginputs: plain[:batch_size],
							  dinputs: plain[:batch_size],
							  keep_prob: 1.0}
					out_G = sess.run([g_out], feed_dict = test_G)
					out_G = (out_G + 1) / 2 * 255
					print(out_G[0])
					save_img(out_G, step)
				saver.save(sess, check_path, global_step = step)

def main():
	predo()
	train()
	# test()

if __name__ == '__main__':
	main()