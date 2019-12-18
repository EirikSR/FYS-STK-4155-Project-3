import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.keras.backend.set_floatx("float64")

def f(x, A):
    #Computes the function from Zhang and Yan article
    I = tf.eye(n,dtype=tf.float64)
    
    a = tf.matmul(tf.transpose(x),x)*A

    b = I - I*tf.matmul(tf.matmul(tf.transpose(x),A),x)
    
    return tf.matmul((a+b),x)

def get_eigval(v, A):
    #Computes eigenvalue from eigenvector and matrix
    v = v.reshape(n,1)    
    a = np.matmul(np.matmul(v.transpose(),A),v)
    b = np.matmul(v.transpose(),v)
    w = np.max(a/b) #Returns a single element in list
    return w

def NN(A, v0, n, t_max, steps, learn_rate, cutoff, epoch, max_ = True):
	#Initializing datasets and converting to tensor

	t = np.linspace(0, t_max, steps)
	x = np.linspace(1, n, n)

	X, T = np.meshgrid(x, t)
	V, T_ = np.meshgrid(v0, t)

	t = T.ravel().reshape(-1,1)
	x = X.ravel().reshape(-1,1)
	v0 = V.ravel().reshape(-1,1)

	x = tf.convert_to_tensor(x)
	t = tf.convert_to_tensor(t)
	v0 = tf.convert_to_tensor(v0)

	points = tf.concat([x, t], 1)

	#Setting layer parameters
	neurons = [50, 25, 10]
	name = [1, 2, 3]

	with tf.name_scope('dnn'):
		#Creating layers
		#Input layer
		former_layer = points

		#Hidden layers
		for layer, nm in zip(neurons, name):
			active_layer = tf.layers.dense(former_layer, layer, name='hidden%d' %nm,\
											activation=tf.nn.sigmoid)
			former_layer = active_layer

		#Output layer
		output = tf.layers.dense(former_layer, 1, name='output')



	with tf.name_scope('cost'):
		#Checking for max or min eigenvalue, initial trial solution
		if max_:
			trial = (output*t + v0) 
		else:
			trial = (output*t - v0) #Doesn't do much
			print("false")

		#Calculating gradient
		trial_dt = tf.gradients(trial, t)

		trial = tf.reshape(trial, (steps, n))
		trial_dt = tf.reshape(trial_dt, (steps, n))

		cost = 0

		for i in range(steps):

			x_ = tf.reshape(trial[i], (n,1))
			x_dt = tf.reshape(trial_dt[i], (n,1))

			dx_dt = f(x_, A) - x_ #equation 1, Zhong et al, cost function

			error = tf.square(dx_dt - x_dt) #squared difference
			
			cost += tf.reduce_sum(error)

		cost = tf.reduce_sum(cost/(n*steps), name='cost')

	with tf.name_scope('trian'):

		optimizer = tf.train.AdamOptimizer(learn_rate)
		trainer = optimizer.minimize(cost)


	init = tf.global_variables_initializer()

	with tf.Session() as S:
		#Initializing the model
		init.run()

		#Training the neural network
		for i in range(epoch):

			S.run(trainer)

			#Printing cost each 1000 step
			if(i % 1000 == 0):
				print("Epoch: ", i, " Cost: ", cost.eval())
				if (cost.eval() < cutoff):
					#Stopping the training if cost is lower than threshold
					break

		print("Final cost: ", cost.eval())

		#Returning final eigenvector
		result = tf.reshape(trial, (steps, n))
		result = result.eval()

		return result, t

tf.set_random_seed(131)
np.random.seed(121)

#Setting up problem parameters
n = 6 
t_max = 1
dt = 0.025
steps = int(t_max/dt)

#Model prameters
cutoff = 1e-4
learn_rate = 0.001 
epoch = 40000
max_ = True
v0 = np.random.rand(n)

#Creating matrix
r = np.random.rand(n,n)
A = (r.T+r)/2.
A_np = A
if (max_ == False):
	A = -A


A = tf.convert_to_tensor(A, dtype=tf.float64)

result, t_= NN(A, v0, n, t_max, steps, learn_rate, cutoff, epoch, max_)

#print(result)

w, v = np.linalg.eig(A_np)
idx = np.argsort(w)
v_np = v[:,idx]
np_w_min = np.min(w)
np_w_max = np.max(w)
np_v_min= v_np[:,0]
np_v_max = v_np[:,-1]

print("Numpy max Eigenvector: ", np_v_max)
print("Numpy min Eigenvector: ", np_v_min)
print("Numpy max Eigenvalue: ", np_w_max)
print("Numpy min Eigenvalue: ", np_w_min)
print("Vectors:", v )



w_nn = result[-1]
v_nn = get_eigval(w_nn, A_np)
print("Neural Network computed Eigenvalue: ", v_nn)
print("Corresponting Eigenvector: ", w_nn)
lst = []
np_plot = []
for vec in result:
	lst.append(get_eigval(vec, A_np))

	if (max_ == True):
		np_plot.append(np_w_max)

	else:
		np_plot.append(np_w_min)




#Plotting results eigenvalue convergence

t = np.linspace(0, t_max, steps)

plt.plot(t,lst)
plt.plot(t,np_plot)
plt.plot()
plt.title("Eigenvalue convergence")
plt.grid(color='grey', linestyle='dashed', linewidth=0.5)
plt.legend(["Neural Network", "Numpy"])
plt.xlabel("Time")
plt.ylabel("Eigenvalue")
plt.show()


#Plotting eigenvector convergence
lst = result.T

for x in lst:
	plt.plot(t, x)


plt.title("Eigenvector convergence")
plt.grid(color='grey', linestyle='dashed', linewidth=0.5)
plt.legend(["Neural Network"])
plt.xlabel("Time")
plt.ylabel("Eigenvector component value")
plt.show()

#Printing corresponding matrix
print(A_np)