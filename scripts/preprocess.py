import numpy as np
from sklearn.decomposition import PCA


def standardize(x, with_ones = False):
	mask = (x != -999)

	# compute the mean and standard deviations
	mean = (x * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))
	# print(std_dev)
	# ------- standarization finish ------------
	stand_x = (x * mask - mean)/std_dev

	# --------- Setting -999 to 0 --------------
	stand_x[~mask] = 0

	# ---------- Add ones to the matrix --------
	if with_ones:
		tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
		tmp[:,1:] = stand_x
		stand_x = tmp

	return stand_x

def getPCA(x, num):
    pca = PCA(n_components = num)
    x = pca.fit_transform(x)
    return x


def standardize_with_power_2(x, with_ones = False):
	mask = (x != -999)

	x_sq = x
	x_sq = x_sq * x_sq
	
	mean = (x_sq * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x_sq - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))

	x_sq = (x_sq * mask - mean)/std_dev
	x_sq[~mask] = 0

	# compute the mean and standard deviations
	mean = (x * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))
	# print(std_dev)
	# ------- standarization finish ------------
	stand_x = (x * mask - mean)/std_dev

	# --------- Setting -999 to 0 --------------
	stand_x[~mask] = 0

	stand_x = np.concatenate((stand_x,x_sq),axis = 1)

	# ---------- Add ones to the matrix --------
	if with_ones:
		tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
		tmp[:,1:] = stand_x
		stand_x = tmp

	return stand_x

def standardize_with_all_power_2(x, with_ones = False):
	mask = (x != -999)

	# compute the mean and standard deviations
	mean = (x * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))
	# print(std_dev)
	# ------- standarization finish ------------
	stand_x = (x * mask - mean)/std_dev

	# --------- Setting -999 to 0 --------------
	stand_x[~mask] = 0
	

	x_cr = np.zeros([x.shape[0], (int)(x.shape[1] * (x.shape[1] + 1) / 2)])
	
	print(x.shape[1])
	print((int)(x.shape[1] * (x.shape[1] - 1) / 2))

	glob_count = 0
	for i in range(x.shape[1]):
		for j in range(i, x.shape[1]):
			x_cr[:, glob_count] = stand_x[:, i] * stand_x[:, j]
			glob_count += 1

	print(glob_count)
	stand_x = np.concatenate((stand_x,x_cr),axis = 1)

	# ---------- Add ones to the matrix --------
	if with_ones:
		tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
		tmp[:,1:] = stand_x
		stand_x = tmp

	return stand_x


def standardize_with_power_23(x, with_ones = False):
	mask = (x != -999)

	x_sq = x
	x_sq = x_sq * x_sq
	
	mean = (x_sq * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x_sq - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))

	x_sq = (x_sq * mask - mean)/std_dev
	x_sq[~mask] = 0


	x_cu = x
	x_cu = x_cu * x_cu * x_cu
	
	mean = (x_cu * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x_cu - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))

	x_cu = (x_cu * mask - mean)/std_dev
	x_cu[~mask] = 0

	# compute the mean and standard deviations
	mean = (x * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))
	# print(std_dev)
	# ------- standarization finish ------------
	stand_x = (x * mask - mean)/std_dev

	# --------- Setting -999 to 0 --------------
	stand_x[~mask] = 0

	stand_x = np.concatenate((stand_x,x_sq),axis = 1)
	stand_x = np.concatenate((stand_x,x_cu),axis = 1)

	# ---------- Add ones to the matrix --------
	if with_ones:
		tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
		tmp[:,1:] = stand_x
		stand_x = tmp

	return stand_x


def standardize_with_power_terms(x, power, with_ones = True):
	mask = (x != -999)

	
	# compute the mean and standard deviations
	mean = (x * mask).sum(axis=0)/np.sum(mask, axis=0)
	std_dev = np.sqrt((((x - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))
	# print(std_dev)
	# ------- standarization finish ------------
	stand_x = (x * mask - mean)/std_dev

	# --------- Setting -999 to 0 --------------
	stand_x[~mask] = 0


	x_tmp = x
	for deg in range(2, power + 1):
		x_tmp = x_tmp * x
		x_sq = x_tmp

		mean = (x_sq * mask).sum(axis=0)/np.sum(mask, axis=0)
		std_dev = np.sqrt((((x_sq - mean) * mask)**2).sum(axis=0)/np.sum(mask, axis=0))

		x_sq = (x_sq * mask - mean)/std_dev
		x_sq[~mask] = 0
		stand_x = np.concatenate((stand_x,x_sq),axis = 1)

	# ---------- Add ones to the matrix --------
	if with_ones:
		tmp = np.ones([stand_x.shape[0], stand_x.shape[1] + 1])
		tmp[:,1:] = stand_x
		stand_x = tmp

	return stand_x