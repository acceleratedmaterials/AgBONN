import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from GPyOpt.methods import BayesianOptimization

#%% load the data

raw_input = pd.read_excel('experimental_data.xlsx','input')

raw_spectra_t = pd.read_excel('experimental_data.xlsx','spectra',index_col=0)


raw_target = pd.read_excel('experimental_data.xlsx','target')

#intepolate the traget spectra to have the same inteval as the measured spectra
import scipy.interpolate as interp

wave = np.arange(380,801,1)

f = interp.interp1d(raw_target.iloc[:,0],raw_target.iloc[:,2],kind='slinear')

target_spec = f(wave)

scaler = MinMaxScaler()

target_spec_norm = scaler.fit_transform(target_spec.reshape(-1,1))





#%% define the loss function

def step_int(x):
    if x>1.2:
        y = 0
    elif 0.7<=x<=1.2:
        y = 1
    elif 0<x<0.7:
        y = x/0.7 
    return y



def spectra_loss_function (spectra, target_spec_norm):
    data = spectra.values
    
    loss = []
   
    
    for i in range(data.shape[1]):
        step_coeff = step_int(max(data[:,i]))
        data_col = scaler.fit_transform(data[:,i].reshape(-1,1))
        cos_loss = cosine_similarity(target_spec_norm.T,data_col.T)
        
        single_loss = cos_loss*step_coeff
        loss.append(single_loss[0])
    loss=np.array(loss)   
 
    return loss



loss  = spectra_loss_function (raw_spectra_t, target_spec_norm)
loss1 = 1-loss

#%%plot the loss evolution

X_init_1 = raw_input.values
Y_init_1 = 1-loss

scaler = MinMaxScaler()
X = X_init_1
X= X[:,3:8]


raw_input['loss'] = Y_init_1


plot_input_all = raw_input
ax = sns.boxplot(x='Run ID',y='loss',hue='method',data=plot_input_all)
ax.set_ylim([0,1])



#%%
    
X_init= X_init_1[:,3:]



#%% define constraints for BO
bds = [  {'name':'QAgNO3','type':'continuous','domain':(0.5,80)},
         {'name':'Qpva','type':'continuous','domain':(10,40)},
          {'name':'QNaOH','type':'continuous','domain':(0.5,80)},
          {'name':'Qhydra','type':'continuous','domain':(0.5,80)},
          {'name':'Qoil','type':'continuous','domain':(100,1000)},
             
        ]

constraints = [
    {
        'name': 'constr_1',
        'constraint': 'x[:,0] + x[:,1] + x[:,2] + x[:,3] - 90'
    },

     
    
]


#%% perform Bayesian optimization
batch_optimizer = BayesianOptimization(f= None,
                                       domain = bds,
                                       constraints = constraints,
                                       model_type='GP',
                                       acquisition_type ='EI',
                                       acquisition_jitter = 0.1,
                                       X=X_init,
                                       Y=Y_init_1,
                                       evaluator_type = 'local_penalization',
                                       batch_size = 1,
                                       num_cores = 16,
                                       minimize= True)

batch_x_next = batch_optimizer.suggest_next_locations()


print(batch_x_next)



cond = np.array(['AgNO3',	'PVA',	'NaOH',	'Hydrazine','Flow rate'
])



np.savetxt('new_conditions.csv',batch_x_next,delimiter=",")



#%%

