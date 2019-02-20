'''
README

Reads the logilfe.txt produced by senet_train.py and plots the losses.
'''

from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt

#with open('logfile.txt', 'r') as inf:
#	lines = inf.readlines()
lines = '''T: 1	 , 1.479171e+00, 4.073592e-02, 1.865133e-01, 2.690543e-01, 2.964200e-01, 3.415800e-01, 3.448677e-01
T: 2	 , 6.467873e-01, 1.900001e-02, 8.072771e-02, 1.220589e-01, 1.286727e-01, 1.460856e-01, 1.502424e-01
T: 3	 , 5.958033e-01, 1.740971e-02, 7.408243e-02, 1.125996e-01, 1.188137e-01, 1.346711e-01, 1.382267e-01
T: 4	 , 5.698129e-01, 1.660098e-02, 7.068686e-02, 1.077676e-01, 1.138500e-01, 1.288795e-01, 1.320279e-01
T: 5	 , 5.528511e-01, 1.608798e-02, 6.850745e-02, 1.046551e-01, 1.105983e-01, 1.250767e-01, 1.279255e-01
V: 5 , 2.936048e+09, 7.415784e+07, 3.847997e+08, 5.515757e+08, 5.941404e+08, 6.551653e+08, 6.762091e+08
T: 6	 , 5.403208e-01, 1.571519e-02, 6.692128e-02, 1.023793e-01, 1.081827e-01, 1.222547e-01, 1.248675e-01
T: 7	 , 5.306591e-01, 1.543481e-02, 6.570644e-02, 1.006223e-01, 1.063155e-01, 1.200746e-01, 1.225054e-01
T: 8	 , 5.224792e-01, 1.519106e-02, 6.466586e-02, 9.912376e-02, 1.047287e-01, 1.182380e-01, 1.205319e-01
T: 9	 , 5.155800e-01, 1.497551e-02, 6.379227e-02, 9.786597e-02, 1.033943e-01, 1.166858e-01, 1.188661e-01
T: 10	 , 5.095111e-01, 1.479512e-02, 6.301937e-02, 9.675126e-02, 1.022188e-01, 1.153274e-01, 1.173991e-01
V: 10 , 3.446637e+10, 5.014665e+09, 6.212359e+09, 5.793354e+09, 5.895483e+09, 5.740689e+09, 5.809825e+09
T: 11	 , 5.923931e+00, 9.744032e-01, 9.858569e-01, 9.882097e-01, 9.905531e-01, 9.918874e-01, 9.930203e-01
T: 12	 , 5.856470e+00, 9.610422e-01, 9.733243e-01, 9.770776e-01, 9.801693e-01, 9.818770e-01, 9.829799e-01
T: 13	 , 5.800023e+00, 9.510971e-01, 9.631488e-01, 9.677613e-01, 9.712109e-01, 9.730168e-01, 9.737887e-01
T: 14	 , 5.751858e+00, 9.423848e-01, 9.546092e-01, 9.599070e-01, 9.635988e-01, 9.654097e-01, 9.659483e-01
T: 15	 , 5.708473e+00, 9.346982e-01, 9.469701e-01, 9.528985e-01, 9.567184e-01, 9.585093e-01, 9.586783e-01
V: 15 , 3.444137e+10, 4.588489e+09, 6.217818e+09, 5.902281e+09, 6.014983e+09, 5.837173e+09, 5.880630e+09
T: 16	 , 5.667456e+00, 9.272874e-01, 9.398639e-01, 9.463130e-01, 9.502423e-01, 9.519556e-01, 9.517942e-01
T: 17	 , 5.633260e+00, 9.211307e-01, 9.339607e-01, 9.408218e-01, 9.448099e-01, 9.464915e-01, 9.460456e-01
T: 18	 , 5.602481e+00, 9.161715e-01, 9.285878e-01, 9.357567e-01, 9.397692e-01, 9.414379e-01, 9.407581e-01
T: 19	 , 5.571774e+00, 9.105509e-01, 9.233855e-01, 9.308995e-01, 9.348731e-01, 9.364817e-01, 9.355834e-01
T: 20	 , 5.542977e+00, 9.058102e-01, 9.185184e-01, 9.262923e-01, 9.301840e-01, 9.316893e-01, 9.304826e-01
V: 20 , 3.453437e+10, 4.961607e+09, 6.250312e+09, 5.828726e+09, 5.916303e+09, 5.760698e+09, 5.816720e+09
T: 21	 , 5.520082e+00, 9.020795e-01, 9.145411e-01, 9.225282e-01, 9.264224e-01, 9.279282e-01, 9.265829e-01
T: 22	 , 5.494233e+00, 8.976595e-01, 9.102689e-01, 9.184999e-01, 9.222562e-01, 9.236020e-01, 9.219465e-01
T: 23	 , 5.472557e+00, 8.937006e-01, 9.066423e-01, 9.150538e-01, 9.187596e-01, 9.200931e-01, 9.183080e-01
T: 24	 , 5.452550e+00, 8.906125e-01, 9.032782e-01, 9.118333e-01, 9.154552e-01, 9.166716e-01, 9.146990e-01
T: 25	 , 5.432824e+00, 8.873171e-01, 8.999306e-01, 9.086784e-01, 9.122238e-01, 9.134045e-01, 9.112696e-01
V: 25 , 3.415932e+10, 4.727528e+09, 6.167902e+09, 5.792381e+09, 5.904289e+09, 5.743960e+09, 5.823258e+09
T: 26	 , 5.416396e+00, 8.845712e-01, 8.971790e-01, 9.060160e-01, 9.095345e-01, 9.106945e-01, 9.084012e-01
T: 27	 , 5.399234e+00, 8.816944e-01, 8.943100e-01, 9.032895e-01, 9.067155e-01, 9.078490e-01, 9.053760e-01
T: 28	 , 5.383367e+00, 8.790451e-01, 8.917388e-01, 9.007933e-01, 9.041175e-01, 9.051377e-01, 9.025344e-01
T: 29	 , 5.368193e+00, 8.768559e-01, 8.891901e-01, 8.983289e-01, 9.015526e-01, 9.025076e-01, 8.997578e-01
T: 30	 , 5.352941e+00, 8.741932e-01, 8.866892e-01, 8.959267e-01, 8.990935e-01, 8.999952e-01, 8.970436e-01
V: 30 , 3.460204e+10, 4.951830e+09, 6.280958e+09, 5.841521e+09, 5.933802e+09, 5.766660e+09, 5.827272e+09
T: 31	 , 5.339717e+00, 8.719689e-01, 8.845007e-01, 8.938281e-01, 8.969263e-01, 8.977825e-01, 8.947108e-01
T: 32	 , 5.326593e+00, 8.701142e-01, 8.823741e-01, 8.917382e-01, 8.947097e-01, 8.954642e-01, 8.921928e-01'''.split('\n')

train_losses = defaultdict(list)
val_losses = defaultdict(list)

for line in lines:
	if line[0] == 'T':
		losses = line.strip().split(',')[1:] # strip whitespace, drop epoch number
		for i, loss in enumerate(losses):
			train_losses[i].append(loss)
	elif line[0] == 'V':
		losses = line.strip().split(',')[1:] # strip whitespace, drop epoch number
		for i, loss in enumerate(losses):
			val_losses[i].append(loss)

new_train = defaultdict(list)
for k, vv in train_losses.items():
	for v in vv:
		new_train[k].append(float(v))

new_val = defaultdict(list)
for k, vv in val_losses.items():
	for v in vv:
		new_val[k].append(float(v))

train_losses = new_train
val_losses = new_val

plt.plot(train_losses[0], c='g')
plt.plot(train_losses[1], c='b')
plt.plot(train_losses[2], c='r')
plt.plot(train_losses[3], c='y')
plt.plot(train_losses[4], c='c')
plt.plot(train_losses[5], c='m')
plt.plot(train_losses[6], c='k')
plt.title('Training Losses (weights from Feature Loss network)')
plt.show()

plt.plot(val_losses[0], c='g')
plt.plot(val_losses[1], c='b')
plt.plot(val_losses[2], c='r')
plt.plot(val_losses[3], c='y')
plt.plot(val_losses[4], c='c')
plt.plot(val_losses[5], c='m')
plt.plot(val_losses[6], c='k')
plt.title('Validation Losses (weights from Feature Loss network)')
plt.show()
