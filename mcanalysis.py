## Analysis of use of historical projects in physics class
## Data file name: mcdata2018.csv
## Answer key: answerkey.csv
## Misconception/naive answer key: misconkey.csv


import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# entering in data into dataframes
df = pd.read_csv("mcdata2018.csv")
key = pd.read_csv("answerkey.csv")
miscon = pd.read_csv("misconkey.csv")


prescore = np.zeros((len(df.index),3))
post1score = np.zeros((len(df.index),3))
post2score = np.zeros((len(df.index),3))


## splitting dataframe into test and control groups
df1, df2, df3 = [x for _, x in df.groupby(df['group'])]

#print(df1) #cont-int
#print(df2) #cont-nat
#print(df3) #test


## matrixes prepared
prescore1 = np.zeros((len(df1.index),3))
post1score1 = np.zeros((len(df1.index),3))
post2score1 = np.zeros((len(df1.index),3))

prescore2 = np.zeros((len(df2.index),3))
post1score2 = np.zeros((len(df2.index),3))
post2score2 = np.zeros((len(df2.index),3))

prescore3 = np.zeros((len(df3.index),3))
post1score3 = np.zeros((len(df3.index),3))
post2score3 = np.zeros((len(df3.index),3))

prematrixtest = np.zeros((3,3))
post1matrixtest = np.zeros((3,3))
post2matrixtest = np.zeros((3,3))

prematrixcontint = np.zeros((3,3))
post1matrixcontint = np.zeros((3,3))
post2matrixcontint = np.zeros((3,3))

prematrixcontnat = np.zeros((3,3))
post1matrixcontnat = np.zeros((3,3))
post2matrixcontnat = np.zeros((3,3))


## pretest scoring
for i in range (0,len(df.index)):
	for j in range (0,4):
		if df.iat[i,j+1] == key.iat[0,j]:
			prescore[i,0] += 1
		elif df.iat[i,j+1] == miscon.iat[0,j]:
			prescore[i,1] += 1
		else:
			prescore[i,2] += 1

for i in range (0,len(df1.index)):
	for j in range (0,4):
		if df1.iat[i,j+1] == key.iat[0,j]:
			prescore1[i,0] += 1
		elif df1.iat[i,j+1] == miscon.iat[0,j]:
			prescore1[i,1] += 1
		else:
			prescore1[i,2] += 1

for i in range (0,len(df2.index)):
	for j in range (0,4):
		if df2.iat[i,j+1] == key.iat[0,j]:
			prescore2[i,0] += 1
		elif df2.iat[i,j+1] == miscon.iat[0,j]:
			prescore2[i,1] += 1
		else:
			prescore2[i,2] += 1

for i in range (0,len(df3.index)):
	for j in range (0,4):
		if df3.iat[i,j+1] == key.iat[0,j]:
			prescore3[i,0] += 1
		elif df3.iat[i,j+1] == miscon.iat[0,j]:
			prescore3[i,1] += 1
		else:
			prescore3[i,2] += 1


## posttest 1 scoring
for i in range (0,len(df.index)):
	for j in range (0,4):
		if df.iat[i,j+1+len(key.columns)] == key.iat[0,j]:
			post1score[i,0] += 1
		elif df.iat[i,j+1+len(key.columns)] == miscon.iat[0,j]:
			post1score[i,1] += 1
		else:
			post1score[i,2] += 1

for i in range (0,len(df1.index)):
	for j in range (0,4):
		if df1.iat[i,j+1+len(key.columns)] == key.iat[0,j]:
			post1score1[i,0] += 1
		elif df1.iat[i,j+1+len(key.columns)] == miscon.iat[0,j]:
			post1score1[i,1] += 1
		else:
			post1score1[i,2] += 1

for i in range (0,len(df2.index)):
	for j in range (0,4):
		if df2.iat[i,j+1+len(key.columns)] == key.iat[0,j]:
			post1score2[i,0] += 1
		elif df2.iat[i,j+1+len(key.columns)] == miscon.iat[0,j]:
			post1score2[i,1] += 1
		else:
			post1score2[i,2] += 1

for i in range (0,len(df3.index)):
	for j in range (0,4):
		if df3.iat[i,j+1+len(key.columns)] == key.iat[0,j]:
			post1score3[i,0] += 1
		elif df3.iat[i,j+1+len(key.columns)] == miscon.iat[0,j]:
			post1score3[i,1] += 1
		else:
			post1score3[i,2] += 1


## posttest 2 (delayed posttest) scoring
### all
for i in range (0,len(df.index)):
	for j in range (0,4):
		if df.iat[i,j+1+2*len(key.columns)] == key.iat[0,j]:
			post2score[i,0] += 1
		elif df.iat[i,j+1+2*len(key.columns)] == miscon.iat[0,j]:
			post2score[i,1] += 1
		else:
			post2score[i,2] += 1

### cont-int
for i in range (0,len(df1.index)):
	for j in range (0,4):
		if df1.iat[i,j+1+2*len(key.columns)] == key.iat[0,j]:
			post2score1[i,0] += 1
		elif df1.iat[i,j+1+2*len(key.columns)] == miscon.iat[0,j]:
			post2score1[i,1] += 1
		else:
			post2score1[i,2] += 1

### cont-nat
for i in range (0,len(df2.index)):
	for j in range (0,4):
		if df2.iat[i,j+1+2*len(key.columns)] == key.iat[0,j]:
			post2score2[i,0] += 1
		elif df2.iat[i,j+1+2*len(key.columns)] == miscon.iat[0,j]:
			post2score2[i,1] += 1
		else:
			post2score2[i,2] += 1

### test
for i in range (0,len(df3.index)):
	for j in range (0,4):
		if df3.iat[i,j+1+2*len(key.columns)] == key.iat[0,j]:
			post2score3[i,0] += 1
		elif df3.iat[i,j+1+2*len(key.columns)] == miscon.iat[0,j]:
			post2score3[i,1] += 1
		else:
			post2score3[i,2] += 1



'''
presum = sum(prescore)
post1sum = sum(post1score)
post2sum = sum(post2score)

normgain1 = (post1sum[0] - presum[0])/(len(key.columns)*len(df.index)-presum[0])
normgain2 = (post2sum[0] - presum[0])/(len(key.columns)*len(df.index)-presum[0])

print(normgain1)
print(normgain2)
'''

presum1 = sum(prescore1)
presum2 = sum(prescore2)
presum3 = sum(prescore3)
post1sum1 = sum(post1score1)
post1sum2 = sum(post1score2)
post1sum3 = sum(post1score3)
post2sum1 = sum(post2score1)
post2sum2 = sum(post2score2)
post2sum3 = sum(post2score3)

## fill off-diagonal elements of test group matrices
prematrixtest[0,1] = prematrixtest[1,0] = sum(np.sqrt(prescore3[:,0]*prescore3[:,1]))
prematrixtest[0,2] = prematrixtest[2,0] = sum(np.sqrt(prescore3[:,0]*prescore3[:,2]))
prematrixtest[1,2] = prematrixtest[2,1] = sum(np.sqrt(prescore3[:,1]*prescore3[:,2]))
post1matrixtest[0,1] = post1matrixtest[1,0] = sum(np.sqrt(post1score3[:,0]*post1score3[:,1]))
post1matrixtest[0,2] = post1matrixtest[2,0] = sum(np.sqrt(post1score3[:,0]*post1score3[:,2]))
post1matrixtest[1,2] = post1matrixtest[2,1] = sum(np.sqrt(post1score3[:,1]*post1score3[:,2]))
post2matrixtest[0,1] = post2matrixtest[1,0] = sum(np.sqrt(post2score3[:,0]*post2score3[:,1]))
post2matrixtest[0,2] = post2matrixtest[2,0] = sum(np.sqrt(post2score3[:,0]*post2score3[:,2]))
post2matrixtest[1,2] = post2matrixtest[2,1] = sum(np.sqrt(post2score3[:,1]*post2score3[:,2]))
## fill off-diagonal elements of control group-int matrices
prematrixcontint[0,1] = prematrixcontint[1,0] = sum(np.sqrt(prescore1[:,0]*prescore1[:,1]))
prematrixcontint[0,2] = prematrixcontint[2,0] = sum(np.sqrt(prescore1[:,0]*prescore1[:,2]))
prematrixcontint[1,2] = prematrixcontint[2,1] = sum(np.sqrt(prescore1[:,1]*prescore1[:,2]))
post1matrixcontint[0,1] = post1matrixcontint[1,0] = sum(np.sqrt(post1score1[:,0]*post1score1[:,1]))
post1matrixcontint[0,2] = post1matrixcontint[2,0] = sum(np.sqrt(post1score1[:,0]*post1score1[:,2]))
post1matrixcontint[1,2] = post1matrixcontint[2,1] = sum(np.sqrt(post1score1[:,1]*post1score1[:,2]))
post2matrixcontint[0,1] = post2matrixcontint[1,0] = sum(np.sqrt(post2score1[:,0]*post2score1[:,1]))
post2matrixcontint[0,2] = post2matrixcontint[2,0] = sum(np.sqrt(post2score1[:,0]*post2score1[:,2]))
post2matrixcontint[1,2] = post2matrixcontint[2,1] = sum(np.sqrt(post2score1[:,1]*post2score1[:,2]))
## fill off-diagonal elements of control group-nat matrices
prematrixcontnat[0,1] = prematrixcontnat[1,0] = sum(np.sqrt(prescore2[:,0]*prescore2[:,1]))
prematrixcontnat[0,2] = prematrixcontnat[2,0] = sum(np.sqrt(prescore2[:,0]*prescore2[:,2]))
prematrixcontnat[1,2] = prematrixcontnat[2,1] = sum(np.sqrt(prescore2[:,1]*prescore2[:,2]))
post1matrixcontnat[0,1] = post1matrixcontnat[1,0] = sum(np.sqrt(post1score2[:,0]*post1score2[:,1]))
post1matrixcontnat[0,2] = post1matrixcontnat[2,0] = sum(np.sqrt(post1score2[:,0]*post1score2[:,2]))
post1matrixcontnat[1,2] = post1matrixcontnat[2,1] = sum(np.sqrt(post1score2[:,1]*post1score2[:,2]))
post2matrixcontnat[0,1] = post2matrixcontnat[1,0] = sum(np.sqrt(post2score2[:,0]*post2score2[:,1]))
post2matrixcontnat[0,2] = post2matrixcontnat[2,0] = sum(np.sqrt(post2score2[:,0]*post2score2[:,2]))
post2matrixcontnat[1,2] = post2matrixcontnat[2,1] = sum(np.sqrt(post2score2[:,1]*post2score2[:,2]))

## fill diagonal elements of all matrices
for i in range (0,3):
	prematrixcontint[i,i] = presum1[i]
	prematrixcontnat[i,i] = presum2[i]
	prematrixtest[i,i] = presum3[i]
	post1matrixcontint[i,i] = post1sum1[i]
	post1matrixcontnat[i,i] = post1sum2[i]
	post1matrixtest[i,i] = post1sum3[i]
	post2matrixcontint[i,i] = post2sum1[i]
	post2matrixcontnat[i,i] = post2sum2[i]
	post2matrixtest[i,i] = post2sum3[i]

# normalize matrices
prematrixtest = prematrixtest / (len(df3.index) * len(key.columns))
post1matrixtest = post1matrixtest / (len(df3.index) * len(key.columns))
post2matrixtest = post2matrixtest / (len(df3.index) * len(key.columns))
prematrixcontnat = prematrixcontnat / (len(df2.index) * len(key.columns))
post1matrixcontnat = post1matrixcontnat / (len(df2.index) * len(key.columns))
post2matrixcontnat = post2matrixcontnat / (len(df2.index) * len(key.columns))
prematrixcontint = prematrixcontint / (len(df1.index) * len(key.columns))
post1matrixcontint = post1matrixcontint / (len(df1.index) * len(key.columns))
post2matrixcontint = post2matrixcontint / (len(df1.index) * len(key.columns))

## eigenvalues and eigenvectors of all matrices
eigvalpretest, eigvectpretest = LA.eig(prematrixtest)
eigvalpost1test, eigvectpost1test = LA.eig(post1matrixtest)
eigvalpost2test, eigvectpost2test = LA.eig(post1matrixtest)
eigvalprecontint, eigvectprecontint = LA.eig(prematrixcontint)
eigvalpost1contint, eigvectpost1contint = LA.eig(post1matrixcontint)
eigvalpost2contint, eigvectpost2contint = LA.eig(post1matrixcontint)
eigvalprecontint, eigvectprecontnat = LA.eig(prematrixcontnat)
eigvalpost1contnat, eigvectpost1contnat = LA.eig(post1matrixcontnat)
eigvalpost2contnat, eigvectpost2contnat = LA.eig(post1matrixcontnat)


print(eigvalpost2test)
print(eigvectpost2test)

## plotting
'''
labels = ('Correct','Misconception','Random')
y_pos = np.arange(len(labels))

plt.bar(y_pos, presum)
plt.xticks(y_pos, labels)
plt.title('Pretest Scores')
plt.show()

plt.bar(y_pos, post1sum)
plt.xticks(y_pos, labels)
plt.title('Posttest 1 Scores')
plt.show()

plt.bar(y_pos, post2sum)
plt.xticks(y_pos, labels)
plt.title('Posttest 2 Scores')
plt.show()
'''


