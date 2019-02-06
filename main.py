from DCGAN import DCGAN
import tensorflow as tf
import util
import os
import cv2

#Namenjeno za testiranje
def celeb(path):
  X = []
  cnt = 0
  for filename in os.listdir(path):
     img = cv2.imread(path+'\\'+filename)
     img = cv2.resize(img, (64,64))
     cnt += 1
     X.append(img)
     if cnt == 50000:
     	break
  # just loads a list of filenames, we will load them in dynamically
  # because there are many
  dim = 64
  colors = 3

  # for celeb
  d_sizes = {
    'conv_layers': [
      (64, 5, 2, False),
      (128, 5, 2, True),
      (256, 5, 2, True),
      (512, 5, 2, True)
    ],
    'dense_layers': [],
  }
  g_sizes = {
    'z': 100,
    'projection': 512,
    'bn_after_project': True,
    'conv_layers': [
      (256, 5, 2, True),
      (128, 5, 2, True),
      (64, 5, 2, True),
      (colors, 5, 2, False)
    ],
    'dense_layers': [],
    'output_activation': tf.tanh,
  }

  gan = DCGAN(dim, colors, d_sizes, g_sizes)
  gan.fit(X)


if __name__ == '__main__':
	
	if not os.path.exists('samples'):
		os.mkdir('samples')

	celeb("img_align_celeba")

	