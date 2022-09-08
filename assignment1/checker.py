import subprocess
import os
import numpy as np
from numpy import genfromtxt

folders = os.listdir()

#read the true y
true_y = []
for i in range(1,5):
	test = f'./data/Q{i}/test/true_Y.csv'
	if(i == 4):
		true_y.append(genfromtxt(test, delimiter=',',dtype='<U6'))
	else:
		true_y.append(genfromtxt(test, delimiter=','))


def compute(i):
	result = f'./result_{i}.txt'
	if (i != 4):
		stres = genfromtxt(result, delimiter='\n')
	else:
		stres = genfromtxt(result, delimiter='\n',dtype='<U6')
	if(stres.shape != true_y[i-1].shape):
		print(f'result_{i}/txt have {stres.shape[0]} values but expected {true_y[i-1].shape[0]}')
		return
	if (i <= 2):
		err = np.mean((((stres - true_y[i-1]) /true_y[i-1]))**2)
		print("MSE ERROR: ",err)
	else:
		eq = np.sum(stres == true_y[i-1])/true_y[i-1].shape[0]
		print("CORRECT: ", eq)

for i in range(1,5):
	print(f'Checking Q{i}')
	if f'Q{i}' not in folders:
		print(f'No Q{i} found')
		continue
	os.chdir(f'./Q{i}')
	files = os.listdir()
	if f'q{i}.py' in files:
		train = f'../data/Q{i}/train'
		test = f'../data/Q{i}/test'
		if i != 2:
			subprocess.run(['python',f'q{i}.py',train, test])
		else:
			subprocess.run(['python',f'q{i}.py',test])
		files = os.listdir()
		if f'result_{i}.txt' in files:
			accuracy = compute(i)
		else:
			print(f'No result_{i}.txt found in folder Q{i} after train')
	else:
		print(f'No q{i}.py found in folder Q{i}')
	os.chdir('../')
		
	

# print("Checking Q1")


# print(folders)
# print(os.getcwd())
# os.chdir("./Q1")
# print(os.getcwd())