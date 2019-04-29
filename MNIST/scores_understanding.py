import pickle

import matplotlib.pyplot as plt

import numpy as np

score = pickle.load(open("score.pkl", "rb"))

x_val = pickle.load(open("x_val.pkl", "rb"))


y_val = pickle.load(open("y_val.pkl", "rb"))
'''
flag = np.zeros(len(score))
for j in range(200):
    
    for i in score:
        if i[j] == 1:
            flag[j] =flag[j]+1

print (flag)

x = np.where(flag != 0)

r = np.zeros(22)

for i in x:
    r= i


       

plt.plot(r, flag[x])
plt.show()

'''

flag = np.zeros(len(score))
for j in range(200):
    
    for i in score[0:96]:
        if i[j] == 1:
            flag[j] =flag[j]+1

print (flag)

x = np.where(flag != 0)
print (x)

flagy = np.zeros(len(score))
for j in range(200):
    
    for i in score[96:]:
        if i[j] == 1:
            flagy[j] =flagy[j]+1

print (flagy)

x2 = np.where(flagy != 0)
print (x2)




