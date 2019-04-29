from scipy.misc import imsave
import pickle
import numpy as np


score = pickle.load(open("score.pkl", 'rb'))
aa = np.zeros((196, 1))
aa[np.where(score[0][0].astype(int)==1)] = 1
bb = aa.reshape(14,14)
bb = bb * 255
imsave('11.png', bb)