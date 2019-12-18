import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

tf.keras.backend.set_floatx('float64')  # Set default float type
tf.random.set_seed(1)

def analytic (x, t):
	return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def Forward_Euler(X):

	F = 0.5 #Parameters for Forwards Euler
	a = 2.

	dx = X[1]-X[0]
	dt = (dx**2*F)/a #Stability criterion

	len_x = len(X)
	len_y = int(1./dt)

	t = np.linspace(0, 1, int(1./dt))

	arr_2d = np.zeros((len_y, len_x))
	arr_1d = np.zeros(len_x)
	print (arr_2d.shape)
	print (arr_1d.shape)

	U = np.sin(np.pi * X) #Setting initial values
	print (U.shape)
	U[0] = 0.
	U[-1]= 0.

	arr_2d[0] = U

	for i in range(0, len_y):
	
		g = F*(U[0:len_x - 2] - 2*U[1:len_x-1] + U[2:len_x])
		arr_1d[1:len_x-1] = U[1:len_x-1] + g
		arr_1d[0] = 0.
		arr_1d[-1]= 0.

		arr_2d[i]=arr_1d

		U, arr_1d = arr_1d, U

	return arr_2d, arr_1d, X, t

class DNModel (tf.keras.Model):
	def __init__(self):
		super (DNModel, self).__init__()
		self.dense_1 = tf.keras.layers.Dense(150, activation = tf.nn.tanh)
		self.dense_2 = tf.keras.layers.Dense(50, activation = tf.nn.sigmoid)
		self.out = tf.keras.layers.Dense(1, name = "output")

	def call(self, inputs):
		x = self.dense_1(inputs)
		x = self.dense_2(x)
		return self.out(x)

@tf.function
def ic(x):
	#Setting up initial conditions for t=0
	return tf.sin(np.pi * x)

@tf.function
def trial_solution(model, x, t):
	grid = tf.concat([x, t], axis=1)
	return (1-t) * ic(x) + x * (1-x) * t * model(grid)


@tf.function
def loss(model, x, t):
	#Defining the loss function
	with tf.GradientTape(persistent=True) as gt1:
		gt1.watch([x, t])
		with tf.GradientTape(persistent=True) as gt2:
			gt2.watch([x, t])
			dG_dt = trial_solution(model, x, t)

		#Setting up Equation 16 from paper
		dtrail_dt = gt2.gradient(dG_dt, t)
		d2trail_d2x = gt1.gradient(gt2.gradient(dG_dt, x), x)
		#Returning MSE between 0 and the calculated error
	return tf.losses.MSE(tf.zeros_like(d2trail_d2x), d2trail_d2x - dtrail_dt)


@tf.function
def grad(model, x, t):
	print (x)
	with tf.GradientTape() as gt:
		loss_ = loss(model, x, t)
	return loss_, gt.gradient(loss_, model.trainable_variables)		

steps = 101

X = np.linspace(0, 1, steps)
T = np.linspace(0, 1, steps)

#First calculating Euler and comparing it to the analytical
gradient_euler, d1d, p_x, p_t = Forward_Euler(X)
p_x, p_t = np.meshgrid(p_x, p_t)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Forward Euler, dx = 0.01")
ax.plot_surface(p_x, p_t, gradient_euler)

plt.show()

gradient_anlt = analytic(p_x, p_t)

diff = gradient_anlt-gradient_euler

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Forward Euler Difference, dx = 0.1")
ax.plot_surface(p_x, p_t, diff)

plt.show()


#Setting up the model
X, T = np.meshgrid(X, T)

x = X.reshape(-1, 1)
t = T.reshape(-1, 1)

model = DNModel()
optimizer = tf.keras.optimizers.Adam(0.05)

epochs = 1000

#Training the model over 1000 itterations
for e in range (epochs):
	
	cost, gradient = grad(model, x, t)

	optimizer.apply_gradients(zip(gradient, model.trainable_variables))

	if (e % 100 == 0):
		print(e, " Loss: ", tf.math.reduce_mean(cost.numpy()))


gradient_anlt = tf.reshape(analytic(x, t), (steps, steps))
gradient_nn = tf.reshape(trial_solution(model, x, t), (steps, steps))

diff = gradient_anlt-gradient_nn

#Plotting results
def plot(x, t, X, Title):
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	ax.set_title(Title)
	ax.plot_surface(x, t, X)

plot(X, T, gradient_anlt, "Analytic")
plot(X, T, gradient_nn, "Neural Network")
plot(X, T, diff, "Neural Network Difference")
plt.show()

