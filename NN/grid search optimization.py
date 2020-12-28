# -*- coding:utf-8  -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

import time

import math

from scipy.signal import  savgol_filter,chirp, find_peaks, peak_widths

import matplotlib.pylab as pl


# definition of truncated loss

def step_int(x):
    if x>1.2:
        y = 0
    elif 0.7<=x<=1.2:
        y = 1
    elif 0<x<0.7:
        y = x/0.7 
    return y

# load target spectrum
raw_target = pd.read_excel(open('OPNA-Run1-DropletDataAll.xlsx','rb'),'Target')


#interpolate target spectrum for calculation of loss
import scipy.interpolate as interp

wave = np.arange(380,801,1)

f = interp.interp1d(raw_target.iloc[:,0],raw_target.iloc[:,2],kind='slinear')

target_spec = f(wave)

scaler = MinMaxScaler()

target_spec_norm = scaler.fit_transform(target_spec.reshape(-1,1))

# load well trained model
from keras.models import load_model
student_model = load_model('posterior-model1-2-3-4-5-6.h5')


# grid search optimization 
ag=np.linspace(4,45,20)
pva=np.linspace(10,40,20)
tsc=np.linspace(0.5,30,20)
seed=np.linspace(0.5,20,20)
tot=np.linspace(200,1000,20)

spectrum=[]
recipe=[]
count=0
import time
t1=time.time()
for i in ag:
  for j in pva:
    for k in tsc:
      for l in seed:
        for m in tot:
          input=np.array([[float(i),float(j),float(k),float(l),float(m)]])
          limit=float(i)+float(j)+float(k)+float(l)
          data=student_model.predict(input)
          step_coeff = step_int(max(data.flatten()))
          data_col = scaler.fit_transform(data.reshape(-1,1))
          cos_loss = cosine_similarity(target_spec_norm.T,data_col.T)
          single_loss = cos_loss*step_coeff
          loss=1-single_loss.flatten()
          if limit<90:
            #print(count)
            result=np.hstack((input.flatten(),data.flatten(),loss))
            recipe.append(result)
            count+=1       
np.savetxt('recipe_spectrum_loss_posterior6.csv',recipe,delimiter=',')
print(time.time()-t1)












