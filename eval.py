import tensorflow as tf
# from net import build_generator, build_discriminator
from network import build_generator, build_discriminator
import cv2
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

plain = np.zeros((7, 48, 64, 1), float)
real = np.zeros((7, 48, 64, 1), float)
batch_size = 7
check_path = 'model/'

def predo():

	for i in range(7):
		img = cv2.imread('plain/' + str(200+i).zfill(4) + '.png')
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (64, 48))
		plain[i] = np.reshape(img, (48, 64, 1))
		img = cv2.imread('real/' + str(200+i).zfill(4) + '.png')
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.resize(img, (64, 48))
		real[i] = np.reshape(img, (48, 64, 1))
	print("read done...")


def save_img(imgs):
	print("saving img.")
	print(np.shape(imgs))
	dirname = 'output/test'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	for i in range(7):
		cv2.imwrite(dirname + '/' + str(200+i).zfill(4) + '.png', imgs[0][i])


def test():

	global plain, real

	ginputs = tf.placeholder(tf.float32, [batch_size, 48, 64, 1])
	dinputs = tf.placeholder(tf.float32, [batch_size, 48, 64, 1])
	keep_prob = tf.placeholder(tf.float32)
	is_train = tf.placeholder(tf.bool)

	g_out, g_params = build_generator(ginputs, is_train)
	d_out, d_params = build_discriminator(dinputs, g_out, keep_prob)
	d_real = tf.slice(d_out, [0, 0], [batch_size, -1])
	d_fake = tf.slice(d_out, [batch_size, 0], [-1, -1])

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, check_path+"-299")

		plain = plain / 255
		plain = 2 * plain - 1
		real = real / 255
		real = 2 * real - 1
		test_G = {ginputs: plain,
				  dinputs: real,
				  keep_prob: 1.0,
				  is_train: False}
		out_G = sess.run([g_out], feed_dict = test_G)
		out_G[0] = (out_G[0] + 1) / 2 * 255

		print(out_G[0][0])
		save_img(out_G)

def main():
	predo()
	test()

if __name__ == '__main__':
	main()