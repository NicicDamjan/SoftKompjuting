import tensorflow as tf
import numpy as np

class ConvLayer:

	def __init__(self, name, num_of_input_channels, num_of_filters, apply_batch_norm, filter_size = 5, stride = 2, a_f = tf.nn.relu):

		"""
		Konstruktor koji kreaira i inicijalizuje skup parametara W i b na osnovu
		prosledjenih argumenata kao sto su:
		- spacijalne dimenzije filtera
		- broj kanala ulazne slike 
		- ukupan broj filtera koji se koristi u okviru datog sloja
		"""

		self.W = tf.get_variable(
				"W_%s" % name, #Naziv varijable - cvora grafa izracunavanja 
				shape = (filter_size, filter_size, num_of_input_channels, num_of_filters), #filter_size ukazuje na spacijalne dimenzije filtera, num_of_input_channels ukazuje na broj kanala ulazne slike
				#bitan je zato sto se dubina filtera mora poklapati sa dubinom ulaza, i num_of_filters ukazuje na ukupan broj filtera koji zelimo u okviru sloja da koristimop
				initializer = tf.truncated_normal_initializer(stddev = 0.02)
			)

		#Za svaki od filtera imacemo po jedan bias parametar
		self.b = tf.get_variable(
				"b_%s" % name,
				shape = (num_of_filters,),
				initializer = tf.zeros_initializer(),
			)
		self.name = name #Naziv konvolucionog sloja
		self.a_f = a_f #Aktivaciona funkcija koju koristimo u okviru datog sloja
		self.stride = stride #Korak koji koristimo prilikom izvodjenja kovnolucije
		self.apply_batch_norm = apply_batch_norm #boolean, da li primenjujemo batch normalizaciju u okviru sloja ili ne
		self.params = [self.W, self.b] #Parametri koje smo gore definisali

	def forward(self, X, reuse, is_training):
	    
		"""
		Funkcija koja propagira sliku kroz arhitekturu konvolucione mreze
		u cilju izracunavanja gradijenata i gubitka prilikom izvrsavanja
		zadatka modela.

		Argumenti:
			- X: tensor koji predstavlja ulaznu sliku
			- reuse: boolean koji nam govori da li zelimo da iskoristimo 
			skup parametara koji smo inicijalno definisali ili traizmo od
			tensorflow-a da interno napravi novi
			- is_training: boolean koji nam govori da li zelimo da koristimo
			batch normalization ili ne

		Povratna vrednost:
			- a_f(conv_out): skup linearnih aktivacija (aktivaciona mapa)
			koje su filteri u okviru posmatranog sloja uspeli da izazovu.
		"""


		conv_out = tf.nn.conv2d(
			X, #Ulazni signal
			self.W, #Skup parametara, tacnije kolekcija filtera koja odgovara datom konvolucionom sloju
			strides=[1, self.stride, self.stride, 1], #
			padding='SAME' #posto smo padding stavili da ostane isti to znaci da ce izlazni signal biti istih dimenzija kao i ulazni signal
		)
		conv_out = tf.nn.bias_add(conv_out, self.b) #na kraju samo dodamo bias

		if self.apply_batch_norm:
			conv_out = tf.contrib.layers.batch_norm(
				conv_out,
				decay=0.9, 
				updates_collections=None,
				epsilon=1e-5,
				scale=True,
				is_training=is_training,
				reuse=reuse,
				scope=self.name,
			)
		return self.a_f(conv_out) #Nakon sto smo pridobili skup linearnih aktivacija (rezultat operacije konvolucije -conv2d) potrebno je da proverimo na kojim mestima se javlja akcijski potencijal - detector stage


class FractionallyStridedConvLayer:
  
  def __init__(self, name, num_of_input_channels, num_of_filters, output_shape, apply_batch_norm, filter_size = 5, stride = 2, a_f = tf.nn.relu):
   
    self.W = tf.get_variable(
      "W_%s" % name,
      shape=(filter_size, filter_size, num_of_filters, num_of_input_channels), #Uociti da ovde specificiramo num_of_filters kao dimenzije ulaza jer deconvolution od manje slike pravi vecu.
      initializer=tf.random_normal_initializer(stddev=0.02),
    )
    self.b = tf.get_variable(
      "b_%s" % name,
      shape=(num_of_filters,),
      initializer=tf.zeros_initializer(),
    )
    self.name = name
    self.a_f = a_f
    self.stride = stride
    self.apply_batch_norm = apply_batch_norm
    self.output_shape = output_shape
    self.params = [self.W, self.b]

  def forward(self, X, reuse, is_training):
    conv_out = tf.nn.conv2d_transpose(
      value=X,
      filter=self.W,
      output_shape=self.output_shape,
      strides=[1, self.stride, self.stride, 1],
    )
    conv_out = tf.nn.bias_add(conv_out, self.b)

    if self.apply_batch_norm:
      conv_out = tf.contrib.layers.batch_norm(
        conv_out,
        decay=0.9, 
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=is_training,
        reuse=reuse,
        scope=self.name,
      )

    return self.a_f(conv_out)

class DenseLayer(object):
  def __init__(self, name, M1, M2, apply_batch_norm, a_f=tf.nn.relu):
    self.W = tf.get_variable(
      "W_%s" % name,
      shape=(M1, M2),
      initializer=tf.random_normal_initializer(stddev=0.02),
    )
    self.b = tf.get_variable(
      "b_%s" % name,
      shape=(M2,),
      initializer=tf.zeros_initializer(),
    )
    self.a_f = a_f
    self.name = name
    self.apply_batch_norm = apply_batch_norm
    self.params = [self.W, self.b]

  def forward(self, X, reuse, is_training):
    a = tf.matmul(X, self.W) + self.b

    if self.apply_batch_norm:
      a = tf.contrib.layers.batch_norm(
        a,
        decay=0.9, 
        updates_collections=None,
        epsilon=1e-5,
        scale=True,
        is_training=is_training,
        reuse=reuse,
        scope=self.name,
      )
    return self.a_f(a)
