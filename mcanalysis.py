## Analysis of use of historical projects in physics class
## Data file name: mcdata2018.csv
## Answer key: answerkey.csv
## Misconception/naive answer key: misconkey.csv


import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set()

# entering in data into dataframes
df = pd.read_csv("mcdata2018.csv")
key = pd.read_csv("answerkey.csv")
miscon = pd.read_csv("misconkey.csv")


prescore = np.zeros((len(df.index),3))
post1score = np.zeros((len(df.index),3))
post2score = np.zeros((len(df.index),3))


question1 = np.zeros((3,3))
question2 = np.zeros((3,3))
question3 = np.zeros((3,3))
question4 = np.zeros((3,3))


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

## question scoring
## control international students
for j in range (0, len(df1.index)):
	# pretest
	if df1.iat[j,1] == key.iat[0,0]:
		question1[0,0] += 1
	if df1.iat[j,2] == key.iat[0,1]:
		question2[0,0] += 1
	if df1.iat[j,3] == key.iat[0,2]:
		question3[0,0] += 1
	if df1.iat[j,4] == key.iat[0,3]:
		question4[0,0] += 1
	# posttest 1
	if df1.iat[j,1+len(key.columns)] == key.iat[0,0]:
		question1[0,1] += 1
	if df1.iat[j,2+len(key.columns)] == key.iat[0,1]:
		question2[0,1] += 1
	if df1.iat[j,3+len(key.columns)] == key.iat[0,2]:
		question3[0,1] += 1
	if df1.iat[j,4+len(key.columns)] == key.iat[0,3]:
		question4[0,1] += 1
	# posttest 2
	if df1.iat[j,1+2*len(key.columns)] == key.iat[0,0]:
		question1[0,2] += 1
	if df1.iat[j,2+2*len(key.columns)] == key.iat[0,1]:
		question2[0,2] += 1
	if df1.iat[j,3+2*len(key.columns)] == key.iat[0,2]:
		question3[0,2] += 1
	if df1.iat[j,4+2*len(key.columns)] == key.iat[0,3]:
		question4[0,2] += 1
## control national students
for j in range (0, len(df2.index)):
	# pretest
	if df2.iat[j,1] == key.iat[0,0]:
		question1[1,0] += 1
	if df2.iat[j,2] == key.iat[0,1]:
		question2[1,0] += 1
	if df2.iat[j,3] == key.iat[0,2]:
		question3[1,0] += 1
	if df2.iat[j,4] == key.iat[0,3]:
		question4[1,0] += 1
	# posttest 1
	if df2.iat[j,1+len(key.columns)] == key.iat[0,0]:
		question1[1,1] += 1
	if df2.iat[j,2+len(key.columns)] == key.iat[0,1]:
		question2[1,1] += 1
	if df2.iat[j,3+len(key.columns)] == key.iat[0,2]:
		question3[1,1] += 1
	if df2.iat[j,4+len(key.columns)] == key.iat[0,3]:
		question4[1,1] += 1
	# posttest 2
	if df2.iat[j,1+2*len(key.columns)] == key.iat[0,0]:
		question1[1,2] += 1
	if df2.iat[j,2+2*len(key.columns)] == key.iat[0,1]:
		question2[1,2] += 1
	if df2.iat[j,3+2*len(key.columns)] == key.iat[0,2]:
		question3[1,2] += 1
	if df2.iat[j,4+2*len(key.columns)] == key.iat[0,3]:
		question4[1,2] += 1
## test group
for j in range (0, len(df3.index)):
	# pretest
	if df3.iat[j,1] == key.iat[0,0]:
		question1[2,0] += 1
	if df3.iat[j,2] == key.iat[0,1]:
		question2[2,0] += 1
	if df3.iat[j,3] == key.iat[0,2]:
		question3[2,0] += 1
	if df3.iat[j,4] == key.iat[0,3]:
		question4[2,0] += 1
	# posttest 1
	if df3.iat[j,1+len(key.columns)] == key.iat[0,0]:
		question1[2,1] += 1
	if df3.iat[j,2+len(key.columns)] == key.iat[0,1]:
		question2[2,1] += 1
	if df3.iat[j,3+len(key.columns)] == key.iat[0,2]:
		question3[2,1] += 1
	if df3.iat[j,4+len(key.columns)] == key.iat[0,3]:
		question4[2,1] += 1
	# posttest 2
	if df3.iat[j,1+2*len(key.columns)] == key.iat[0,0]:
		question1[2,2] += 1
	if df3.iat[j,2+2*len(key.columns)] == key.iat[0,1]:
		question2[2,2] += 1
	if df3.iat[j,3+2*len(key.columns)] == key.iat[0,2]:
		question3[2,2] += 1
	if df3.iat[j,4+2*len(key.columns)] == key.iat[0,3]:
		question4[2,2] += 1

# make into percentages, divide by numer of students
question1[0,:] = question1[0,:]/len(df1.index)*100
question1[1,:] = question1[1,:]/len(df2.index)*100
question1[2,:] = question1[2,:]/len(df3.index)*100
question2[0,:] = question2[0,:]/len(df1.index)*100
question2[1,:] = question2[1,:]/len(df2.index)*100
question2[2,:] = question2[2,:]/len(df3.index)*100
question3[0,:] = question3[0,:]/len(df1.index)*100
question3[1,:] = question3[1,:]/len(df2.index)*100
question3[2,:] = question3[2,:]/len(df3.index)*100
question4[0,:] = question4[0,:]/len(df1.index)*100
question4[1,:] = question4[1,:]/len(df2.index)*100
question4[2,:] = question4[2,:]/len(df3.index)*100

## plotting bar graphs of individual question correct responses
# Question 1
ind = np.arange(3)
width = 0.2
fig, ax = plt.subplots()
contintval = plt.bar(ind,question1[0,:], width = width, label = 'Control-Inter')
contnatval = plt.bar(ind+width,question1[1,:], width = width, label = 'Control-Nat')
testval = plt.bar(ind+2*width,question1[2,:], width = width, label = 'Test')
plt.legend()
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Pretest', 'Posttest 1', 'Posttest 2') )
plt.title("Question 1 Percent Correct")
#plt.show()
fig.savefig('q1.png')
plt.close(fig)
# Question 2
ind = np.arange(3)
width = 0.2
fig, ax = plt.subplots()
contintval = plt.bar(ind,question2[0,:], width = width, label = 'Control-Inter')
contnatval = plt.bar(ind+width,question2[1,:], width = width, label = 'Control-Nat')
testval = plt.bar(ind+2*width,question2[2,:], width = width, label = 'Test')
plt.legend()
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Pretest', 'Posttest 1', 'Posttest 2') )
plt.title("Question 2 Percent Correct")
#plt.show()
fig.savefig('q2.png')
plt.close(fig)
# Question 3
ind = np.arange(3)
width = 0.2
fig, ax = plt.subplots()
contintval = plt.bar(ind,question3[0,:], width = width, label = 'Control-Inter')
contnatval = plt.bar(ind+width,question3[1,:], width = width, label = 'Control-Nat')
testval = plt.bar(ind+2*width,question3[2,:], width = width, label = 'Test')
plt.legend()
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Pretest', 'Posttest 1', 'Posttest 2') )
plt.title("Question 3 Percent Correct")
#plt.show()
fig.savefig('q3.png')
plt.close(fig)
# Question 4
ind = np.arange(3)
width = 0.2
fig, ax = plt.subplots()
contintval = plt.bar(ind,question4[0,:], width = width, label = 'Control-Inter')
contnatval = plt.bar(ind+width,question4[1,:], width = width, label = 'Control-Nat')
testval = plt.bar(ind+2*width,question4[2,:], width = width, label = 'Test')
plt.legend()
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Pretest', 'Posttest 1', 'Posttest 2') )
plt.title("Question 4 Percent Correct")
#plt.show()
fig.savefig('q4.png')
plt.close(fig)

#['pre','post1','post2']

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
eigvalpost2test, eigvectpost2test = LA.eig(post2matrixtest)
eigvalprecontint, eigvectprecontint = LA.eig(prematrixcontint)
eigvalpost1contint, eigvectpost1contint = LA.eig(post1matrixcontint)
eigvalpost2contint, eigvectpost2contint = LA.eig(post2matrixcontint)
eigvalprecontnat, eigvectprecontnat = LA.eig(prematrixcontnat)
eigvalpost1contnat, eigvectpost1contnat = LA.eig(post1matrixcontnat)
eigvalpost2contnat, eigvectpost2contnat = LA.eig(post2matrixcontnat)


## test model data points
## y-values for Newtonian thiking
## x-values for naieve thinking
y_test = [eigvalpretest[0]**2*eigvectpretest[0,0]**2, eigvalpost1test[0]**2*eigvectpost1test[0,0]**2, eigvalpost2test[0]**2*eigvectpost2test[0,0]**2]
x_test = [eigvalpretest[0]**2*eigvectpretest[1,0]**2, eigvalpost1test[0]**2*eigvectpost1test[1,0]**2, eigvalpost2test[0]**2*eigvectpost2test[1,0]**2]
y_contint = [eigvalprecontint[0]**2*eigvectprecontint[0,0]**2, eigvalpost1contint[0]**2*eigvectpost1contint[0,0]**2, eigvalpost2contint[0]**2*eigvectpost2contint[0,0]**2]
x_contint = [eigvalprecontint[0]**2*eigvectprecontint[1,0]**2, eigvalpost1contint[0]**2*eigvectpost1contint[1,0]**2, eigvalpost2contint[0]**2*eigvectpost2contint[1,0]**2]
y_contnat = [eigvalprecontnat[0]**2*eigvectprecontnat[0,0]**2, eigvalpost1contnat[0]**2*eigvectpost1contnat[0,0]**2, eigvalpost2contnat[0]**2*eigvectpost2contnat[0,0]**2]
x_contnat = [eigvalprecontnat[0]**2*eigvectprecontnat[1,0]**2, eigvalpost1contnat[0]**2*eigvectpost1contnat[1,0]**2, eigvalpost2contnat[0]**2*eigvectpost2contnat[1,0]**2]



## plotting model analysis
fig, ax = plt.subplots()
plt.scatter(x_test,y_test)
plt.plot(x_test,y_test, label = 'Test Group')
plt.scatter(x_contnat,y_contnat)
plt.plot(x_contnat,y_contnat, label = 'Control National')
plt.scatter(x_contint,y_contint)
plt.plot(x_contint,y_contint, label = 'Control International')
plt.plot([0,1],[1,0], color = 'black', lw = 2)
plt.xlim((0,1))
plt.ylim((0,1))
plt.title("Model Analysis of Student's Force & Motion Conceptions")
plt.xlabel('Misconception Probability')
plt.ylabel('Newtonian Probability')
plt.legend()
#plt.show()
fig.savefig('modeling.png')
plt.close(fig)