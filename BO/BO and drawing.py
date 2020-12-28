# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:29:01 2019

@author: Danny
"""


# -*- coding: utf-8 -*-

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

import seaborn as sns
from scipy.signal import  savgol_filter,chirp, find_peaks, peak_widths

import matplotlib.pylab as pl
import matplotlib.animation as animation
import GPy


raw_input_1 = pd.read_excel('GSNA-Run1.xlsx','Sheet1')

raw_spectra = pd.read_excel('GSNA-Run1.xlsx','Sheet2',index_col=0)
raw_target = pd.read_excel('OPNA-Run1.xlsx','Target')

raw_input_1.columns
#drop off columns that we don't have control over or what is measured
cols = [0,1,2]
raw_input = raw_input_1.drop(raw_input_1.columns[[0,1,2]],axis=1)


raw_input.columns

raw_spectra = raw_spectra.iloc[:421,:]

import scipy.interpolate as interp

wave = np.arange(380,801,1)

f = interp.interp1d(raw_target.iloc[:,0],raw_target.iloc[:,2],kind='slinear')

target_spec = f(wave)

scaler = MinMaxScaler()

target_spec_norm = scaler.fit_transform(target_spec.reshape(-1,1))


raw_input_2 = pd.read_excel('GSNA-Run5.xlsx','Sheet1')

#raw_input_2.iloc[:,1] = raw_input_2.iloc[:,1]+15

raw_spectra_2 = pd.read_excel('GSNA-Run5.xlsx','Sheet2',index_col=0)

raw_input = pd.concat([raw_input_1,raw_input_2],axis=0)

raw_spectra_t =  pd.concat([raw_spectra,raw_spectra_2],axis=1)

raw_input_3 = pd.read_excel('GSNA Random run2.xlsx','Sheet1')
raw_spectra_3 = pd.read_excel('GSNA Random run2.xlsx','Sheet2',index_col=0)




#plt.plot(target_spec)
#function that smoothen the output spectrum, extract FWHM, peak location, area under the unwanted peak

#spectra = raw_spectra

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

a = np.argmin(loss1)
plt.plot(raw_spectra_t.iloc[:,a])
plt.plot(wave,target_spec)
print(raw_input.iloc[a,:])
#%%
#
#plt.boxplot(loss2)
##plt.scatter(raw_input_1.iloc[:,1],loss)
#plt_data = pd.DataFrame(loss,columns=['combined_loss'])
#plt_data['Exp_ID'] = raw_input_1.iloc[:,1]
#
#plt_data_peak = plt_data.iloc[:1905,]
#boxplot = plt_data.boxplot(by = 'Exp_ID')
#
#plt.title("Boxplot of combined loss")
## get rid of the automatic 'Boxplot grouped by group_by_column_name' title
#plt.suptitle("")
#
#plt.show()
#plt.plot(loss)
#plt.title("combined loss")
#X_init = raw_input.values
#Y_init = loss

X_init_1 = raw_input.values
Y_init_1 = 1-loss


scaler = MinMaxScaler()


X = X_init_1
X= X[:,3:8]

X1 = np.empty((len(X),2))
X1[:,0] = X[:,0]/X[:,3]
X1[:,1] = X[:,2]/X[:,0]

X1 = scaler.fit_transform(X1)
X1= np.concatenate((Y_init_1,X1),axis=1)


cor = np.corrcoef(X1.T)

corlabels = ['total Loss','AgNO3/Seed','Tsc/AgNO3']

# plot the heatmap
sns.heatmap(np.abs(cor),annot=True,xticklabels=corlabels,yticklabels=corlabels)

#X_init_com = np.vstack((X_init_1,X_init_2))
#
#Y_init_com = np.vstack((Y_init_1,Y_init_2))

#aaa =np.concatenate((Y_init_1,X_init_1[:,3:]),axis=1)
#
#cor = np.corrcoef(aaa.T)
#
#corlabels = ['loss','AgNO3','Pva','Tsc','Seed','Tot']
#
## plot the heatmap
#sns.heatmap(cor,annot=True,xticklabels=corlabels,yticklabels=corlabels)

raw_input['loss'] = Y_init_1
plot_input= raw_input.iloc[:1990+314,:]
plot_input['method'] = 'BO'

plot_input2 = raw_input.iloc[2275+315:,:]
plot_input2['Run ID']=plot_input2['Run ID'].replace([10,11,12],[6,7,8])
plot_input2['method'] = 'NN'

plot_input_all =pd.concat([plot_input,plot_input2])


plt.rcParams["figure.figsize"] = [12, 8]
ax = sns.boxplot(x='Run ID',y='loss',hue='method',data=plot_input_all)
ax.set_ylim([0,1])



#ax1 = sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data = plot_input_all)
#plt.scatter(min_con[7,3],min_con[7,6],500,marker='*',c='r',label='BO best')
#plt.scatter(min_con[11,3],min_con[11,6],500,marker='o',c='r',label = 'NN best')
#plt.legend()
#tips = sns.load_dataset("tips")

#aaa=pd.DataFrame(aaa)
##aaa.to_excel('loss_Ag.xlsx')
#
#ax=aaa.boxplot(column =[8], by=[1])

#bbb = aaa.iloc[:1366,:]

#plt.rcParams["figure.figsize"] = [8, 6]
#plt.rcParams.update({'font.size': 16})
#ax=aaa.boxplot(column =[8], by=[0])
#_ = aaa.plot.scatter(x=[0],y=[8],ax=ax,marker='o',s=5)
#%%
#data_t= aaa.values
Y_new = []
X_new = []

for i in range(int(X_init_1[-1,1])):
    idx = np.where(X_init_1[:,1]==i+1)
    Y_ = np.min(Y_init_1[idx])
    X_ = X_init_1[idx[0][1]]
    Y_new.append(Y_)
    X_new.append(X_)

colors = pl.cm.plasma(np.linspace(0,0.6,5))



plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams.update({'font.size': 20})

new_con = []
idx_con = []
com_all =np.concatenate((X_init_1,Y_init_1),axis=1)

mean_con = []
mini_con = []
for i in range(180):
    idx_con = np.where(com_all[:,1]==i+1)
    loss_con = np.mean(Y_init_1[idx_con])
    loss_con_min = np.min(Y_init_1[idx_con])
    mean_con1 = np.concatenate((X_init_1[idx_con[0][0],:],np.array([loss_con])),axis=0)
    mean_con1 = mean_con1.reshape(1,-1)
    mean_con2 = np.concatenate((X_init_1[idx_con[0][0],:],np.array([loss_con_min])),axis=0)
    mean_con2 = mean_con2.reshape(1,-1)
    mean_con.append(mean_con1)
    mini_con.append(mean_con2)
mean_con = np.vstack(mean_con)
mini_con = np.vstack(mini_con)





from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri

#ax = fig.add_subplot(111, projection='3d')

#
#ax.scatter(mean_con[:119,3],mean_con[:119,6],mean_con[:119,8],label='BO')#)
#ax.scatter(mean_con[135:,3],mean_con[135:,6],mean_con[135:,8],label='NN')#)

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 16})
plt.rcParams["legend.loc"]= 'upper right'
plt.tricontourf(mean_con[:,3],mean_con[:,6],mean_con[:,8],np.linspace(0,1,33))
plt.xlim(5,45)
plt.ylim(0.5,20)
plt.xlabel('QAgNO3(%)')
plt.ylabel('Qseed(%)')
ax = sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data =plot_input_all )



plt.title('Run8')
plt.colorbar()
#%%
#plt.colorbar()
plot_arr = plot_input_all.values

df_mean = pd.DataFrame(mean_con, columns= ['Run ID', 'Condition ID', 'Droplet ID', 'QAgNO3(%)', 'Qpva(%)',
       'Qtsc(%)', 'Qseed(%)', 'Qtot(uL/min)', 'loss'])
    


#df_mean['method'] ='BO'
#df_mean.iloc[120:,:]=df_mean.iloc[120:,:].replace('BO','NN')    
df_mean['Run ID']=df_mean['Run ID'].replace([10,11,12],[6,7,8])

mean_con = df_mean.values

# plt.rcParams
fig1, axs = plt.subplots(2, 4,  figsize= (16,8), constrained_layout=True)

axs = axs.flatten()
for i in range(8):
    # fig = plt.figure()
    idx2 = np.where(mean_con[:,0]<=i+1)
    axs[i].tricontour(np.squeeze(mean_con[idx2,3]),np.squeeze(mean_con[idx2,6]),np.squeeze(mean_con[idx2,8]),np.linspace(0,1,10), linestyles='dashed',colors='white')
    im=axs[i].tricontourf(np.squeeze(mean_con[idx2,3]),np.squeeze(mean_con[idx2,6]),np.squeeze(mean_con[idx2,8]),cmap='plasma')
   
    # plt.colorbar(axs[i])
    axs[i].set_xlim(4,45)
    axs[i].set_ylim(0,20)
    idx = np.where(plot_arr[:,0]<=i+1)
    data = plot_input_all.iloc[idx] #select data range
    p =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data = data, ax=axs[i])
    # idx1 = np.where(plot_arr[:,0]<i+1)
    # data1 = plot_input_all.iloc[idx2] #select data range
    # p1 =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',color=".5",data = data1)
    axs[i].set_title('Run'+str(i+1))

fig1.colorbar(im, ax=axs[3])
fig1.colorbar(im, ax=axs[7])

df1 = df_mean.iloc[:120,3:-1]
df1.to_csv('BO_condition.csv')

#%%

fig1.savefig('manifold', dpi=600)
#ani = animation.FuncAnimation(fig, animate, interval=200)


ax1 = sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data = plot_input_all)




#plt.scatter(min_con[7,3],min_con[7,6],500,marker='*',c='r',label='BO best')
#plt.scatter(min_con[11,3],min_con[11,6],500,marker='o',c='r',label = 'NN best')




from sklearn import decomposition
'''''
PCA

'''

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(np.log(mean_con[:,3:8]))
pca= decomposition.PCA( )
pca_con= pca.fit_transform(x)

e_values = pca.explained_variance_ratio_    
e_vec = pca.components_

k_pca= decomposition.KernelPCA(kernel='')
pca_con= pca.fit_transform(x)

pca_con = np.concatenate((pca_con,np.array(mean_con[:,[0,-1]])),axis=1)

df_pca = pd.DataFrame(pca_con, columns= ['Comp_1', 'Comp_2', 'Comp_3', 'Comp_4', 'Comp_5',
       'Run', 'loss'])


ax = sns.scatterplot(x='Comp_1',y='Comp_2',data = df_pca)
    
ax.legend()
df_pca['method'] ='BO'
df_pca.iloc[135:,:]=df_pca.iloc[135:,:].replace('BO','NN')  

plt.tricontourf(df_pca['Comp_1'],df_pca['Comp_2'],df_pca['loss'])

plt.colorbar()
ax1 = sns.scatterplot(x='Comp_1',y='Comp_2',hue='method',data = df_pca)


'''



'''
idx_con = []
for i in range(12):
    idx = np.where(X_init_1[:,0]==i+1)
    
    idx_plot = np.argsort(Y_init_1[idx],axis=0)
    min_n = idx[0][int(idx_plot[0])]
    med_n = idx[0][int(idx_plot[len(idx_plot)//2])]
    idx_new = np.where(X_init_1[:,1]==X_init_1[min_n,1])
    new_con1 = com_all[idx_new[0]]
    new_con.append(new_con1 )
    idx_con.append(idx_new[0])

new_con= np.vstack(new_con)
new_con= new_con[new_con[:,0]!=9]

fig = plt.figure()

plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.figsize"] = [11, 8]

df_new = pd.DataFrame(new_con, columns= ['Run ID', 'Condition ID', 'Droplet ID', 'QAgNO3(%)', 'Qpva(%)',
       'Qtsc(%)', 'Qseed(%)', 'Qtot(uL/min)', 'loss'])

df_new['method'] ='BO best'
df_new.iloc[155:,:]=df_new.iloc[155:,:].replace('BO best','NN best')    
df_new['Run ID']=df_new['Run ID'].replace([10,11,12],[6,7,8])

df_new['Run ID'] = df_new['Run ID'].astype('int')

ax = sns.stripplot(x='Run ID',y='loss',hue='method',data=plot_input_all,dodge=True)

ax = sns.boxplot(x='Run ID',y='loss',hue='method',data=df_new,palette="Set2")
ax.set_ylim([0,1])

#%%
#fig.savefig('figure2.pdf', format='pdf')  

fig1 = plt.figure()

df_mean['Run ID'] = df_mean['Run ID'].astype('int')

df_mean['method'] ='BO'
df_mean.iloc[135:,:]=df_mean.iloc[135:,:].replace('BO','NN')  

df_mean_p = pd.concat([df_mean.iloc[:120,:],df_mean.iloc[135:,:]])

ax2 = sns.stripplot(x = 'Run ID', y='loss',hue='method',data=df_mean_p,dodge=True)

#ax2.set_xlim([1,8])
ax2.set_ylim([0,1])

#fig1.savefig('figure2.pdf', format='pdf')  

#    X_ = X_init_1[idx[0][1]]
#    Y_new.append(Y_)
#    X_new.append(X_)
new_con = np.asarray(new_con)
#%%
min_con = []
con_loss = []
median_loss = []

plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams.update({'font.size': 22})
for i in range(8):
    idx = np.where(X_init_1[:,0]==i+1)
    
    idx_plot = np.argsort(Y_init_1[idx],axis=0)
    min_n = idx[0][int(idx_plot[0])]
    med_n = idx[0][int(idx_plot[len(idx_plot)//2])]
    min_con.append(X_init_1[min_n,:])
    fig = plt.figure()
    plt.title('Run'+str(i+1))

    aaa = np.array(raw_spectra_t.iloc[:,idx[0][int(idx_plot[0])]])
    bbb=np.array(raw_spectra_t.iloc[:,idx[0][int(idx_plot[len(idx_plot)//2])]])
    aaa=aaa.reshape(-1,1)
    bbb= bbb.reshape(-1,1)
   
#    bbb=scaler.fit_transform(bbb)
#    aaa= scaler.fit_transform(aaa)
#    plt.plot(wave,aaa,label = 'Run'+str(i+1),color= colors[i],linewidth = 3)  
    plt.plot(wave,aaa,label = 'Experiment',linewidth = 3.0, c = 'grey' , linestyle = '--')
#    plt.plot(wave,raw_spectra_t.iloc[:,idx[0][int(idx_plot[100])]],label = 'average')
    #plt.plot(wave,bbb,label = 'Median')
    plt.plot(wave,target_spec_norm,label='Target',linewidth = 3.0, c = 'black')#,color = 'black',linestyle = '--',linewidth = 3)
    plt.legend(loc = 'upperleft')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('normalized intensity [a.u.]')
    con_loss.append(Y_init_1[min_n,:])
    median_loss.append(Y_init_1[med_n,:])
    fig.tight_layout()
    fig.savefig('spec'+str(i))
#    X_ = X_init_1[idx[0][1]]
#    Y_new.append(Y_)
#    X_new.append(X_)

#%%
    
from PIL import Image
import glob
 
# Create the frames
frames = []
imgs = glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('tem.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000, loop=0) 
    
#%%    
plt.rcParams["figure.figsize"] = [9, 8]
plt.rcParams.update({'font.size': 22})
fig = plt.figure()
min_con = np.array(min_con)
con_loss = np.array(con_loss)
median_loss= np.array(median_loss)
Y_new = np.array(Y_new).reshape(-1,1)
X_new = np.array(X_new)




plt.rcParams["legend.loc"]= 'upper right'
ax1= plt.tricontourf(mean_con[:,3],mean_con[:,6],mean_con[:,8],np.linspace(0,1,33),cmap = 'plasma')
plt.tricontour(mean_con[:,3],mean_con[:,6],mean_con[:,8],colors='white',linestyles = 'dashed')
plt.xlim(5,45)
plt.ylim(0.5,20)

ax = sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data =plot_input_all, s=60, linewidth=0.2,edgecolor='silver' )
sns.scatterplot(min_con[[11],3],min_con[11,6],marker ='P', color ='steelblue',s=250,edgecolor='silver' )


sns.scatterplot(min_con[[7],3],min_con[7,6],marker ='P',color ='orange', s=250,edgecolor='silver' )

plt.xlabel('QAgNO3(%)')
plt.ylabel('Qseed(%)')
#ax.get_legend().remove()
plt.yticks([0.5,5,10,15,20])
fig.tight_layout()

fig.savefig('exp_manifold.pdf', format='pdf')  

#%%
    
Y_new = np.array(Y_new).reshape(-1,1)
X_new = np.array(X_new)





#
min(Y_init_1)
X_init= X_init_1[:,3:]

#X_new= X_new[:,3:]
from GPyOpt.methods import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern



m52 = ConstantKernel(1) * Matern(length_scale=1, nu=0.1)
gpr_process = GaussianProcessRegressor(kernel=m52, alpha=0.01)

#gpr_process.fit(df_mean.iloc[:,[7,5]], df_mean.iloc[:,-1])
#mini_con[:,7]= mini_con[:,7]/10
mini_con[:,7] = mini_con[:,7]/50

gpr_process.fit(mini_con[:120,[3,5,6,7]],  df_mean.iloc[:120,-2])

Ag = np.linspace(4,45,40)
pva = np.linspace(10,40,40)
tsc = np.linspace(0.5,30,40)

Seed = np.linspace(0.5,20,40)

tot = np.linspace(200,1000,40)/50
gx, gy,gz,gm = np.meshgrid(Ag, pva,tsc,Seed)


process_mesh= []
ind_mesh= []
for i in range(len(Ag)):
#    for j in range(len(pva)):
    for k in range(len(tsc)):
        for m in range(len(Seed)):
            for n in range(len(tot)):
                b = [Ag[i],tsc[k],Seed[m],tot[n]]
                ind = [i,k,m,n]
            
                process_mesh.append(b)
        
        
process_mesh = np.array(process_mesh) 

gx= process_mesh[:,0].reshape(gx.shape)

gy = process_mesh[:,1].reshape(gx.shape)

gz = process_mesh[:,2].reshape(gx.shape)
gm = process_mesh[:,3].reshape(gx.shape)

kern1 = GPy.kern.Matern52(input_dim=4)
m = GPy.models.GPRegression(mini_con[:120,[3,5,6,7]],  df_mean.iloc[:120,-2].values.reshape(-1,1),kern1)
m.optimize()  
  
mu_process= gpr_process.predict(process_mesh,return_cov= False)


# plt.plot(mu_process)
process_data = np.concatenate((process_mesh,mu_process.reshape(-1,1)),axis=1)
#process_data = process_data.reshape(40,40,40,40,40)




#%%


# create the figure, add a 3d axis, set the viewing angle
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#plt.rcParams.update({'font.size': 12})
#
## here we create the surface plot, but pass V through a colormap
## to create a different color for each patch
##ax.scatter(gx, gy, gz, facecolors=cm.plasma(mu_process))
#mean_p1 = np.mean((mu_process.reshape(gx.shape)),axis=2)
#mean_p2 = np.mean((mu_process.reshape(gx.shape)),axis=1)
#mean_p3 = np.mean((mu_process.reshape(gx.shape)),axis=0)
#
#
#
#xy = ax.contourf(gx[:,:,0], gy[:,:,0],  mean_p1,alpha=0.9,zdir='z',offset=0.5, levels=np.linspace(0,1,33))
#xz= ax.contourf(gx[:,0,:], gz[:,0,:],  mean_p2,zdir='x', offset=10)
#yz= ax.contourf(gx[0,:,:], gy[0,:,:],  mean_p3,alpha=0.9,zdir='y',offset=10, levels=np.linspace(0,1,33))

#sc12 = ax.contourf(X,Y, V, vmin=0, vmax=100, cmap='viridis_r', alpha=0.9, zdir='z', offset=50, levels=np.linspace(0,100,11))
#sc1 = ax.contour(X, Y, V, colors='w', alpha=1, zdir='z', offset=50.1, linewidths=0.3, levels=np.linspace(0,100,11))

#ax.set_xlabel('QAgNO3[%]')
#ax.set_ylabel('Qseed[%]')
#ax.set_zlabel('Qts[%]')
#ax.set_zlim(0.5,30)


#var_process = 1.96 * np.sqrt(np.diag(cov_process))


plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams.update({'font.size': 22})
plt.rcParams["legend.loc"]= 'upper right'
fig = plt.figure()
#plot_input_all.iloc[:,7]=plot_input_all.iloc[:,7]/10
df_gp = pd.DataFrame(process_data, columns= [ 'QAgNO3(%)', 
       'Qtsc(%)', 'Qseed(%)', 'Qtotal(uL/min)', 'loss'])
ppp = df_gp.groupby(['QAgNO3(%)','Qseed(%)'],as_index=False).min()    
plt.contourf(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
             np.array(ppp.iloc[:,-1]).reshape(40,40),np.linspace(0,1,33),cmap='plasma')
plt.contour(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
             np.array(ppp.iloc[:,-1]).reshape(40,40),np.linspace(0,1,30), linestyles='dashed',colors='white')
#plt.contourf(gx, gy, var_process.reshape(gx.shape),33)

# ax1 = sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data = plot_input_all.iloc[:1988+314,:])
plt.ylim(0.5,20)
#plt.xlim(10,40)

plt.xlabel('QAgNO3[%]')
# plt.ylabel('Qseed[%]')
plt.yticks([])

fig.tight_layout()

fig.savefig('BO_manifold.pdf', format='pdf')  

#clb = plt.colorbar()

#%%

mean_con = mean_con [:120,:]
mean_con[:,7] = mean_con[:,7]/50

#%%
i=7
plot_nn_data = plot_input_all.iloc[:1988+314,:]
plot_arr_1 =plot_nn_data.values

#fig1, axs = plt.subplots(2, 4,  figsize= (16,8), constrained_layout=True)
#axs = axs.flatten()
for i in range(8):
#    process_mesh= []
#
#    for k in range(len(Ag)):
#        for j in range(len(pva)):
#    
#            b = [Ag[k],Seed[j]]
#            
#            process_mesh.append(b)
#        
#        
#    process_mesh = np.array(process_mesh) 
#
#    gx= process_mesh[:,0].reshape(len(Ag),len(Ag))
#    
#    gy = process_mesh[:,1].reshape(len(Ag),len(Ag))
    
    idx2 = np.where(mean_con[:,0]<=i+1)
    idx_col = [3,5,6,7]
    X_new = mean_con[idx2[0],:]
    X_new = X_new[:,idx_col]
    gpr_process.fit(X_new, mean_con[idx2[0],-1].reshape(-1,1))
    mu_process = gpr_process.predict(process_mesh, return_cov=False)
    process_data = np.concatenate((process_mesh,mu_process.reshape(-1,1)),axis=1)
#    var_process = 1.96 * np.sqrt(np.diag(cov_process))
#    fig,ax  = plt.subplots(1,1)
    fig = plt.figure()
    df_gp = pd.DataFrame(process_data, columns= [ 'QAgNO3(%)', 
       'Qtsc(%)', 'Qseed(%)','Qtot(uL/min)',  'loss'])
    ppp = df_gp.groupby(['QAgNO3(%)','Qseed(%)'],as_index=False).min()    
    plt.contourf(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
                 np.array(ppp.iloc[:,-1]).reshape(40,40),np.linspace(0,1,100),cmap='plasma')
    
##    if i ==3 or i ==7:
##        axs[i].set_colorbar(ticks=np.linspace(0,1,5))
##    
    plt.contour(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
             np.array(ppp.iloc[:,-1]).reshape(40,40),np.linspace(0,1,30), linestyles='dashed',colors='white')
#    
#
##    p =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',data = data)
#    
##    plt.setp(ax.get_yticklabels(), visible=False)
#    
#    axs[i].set_xlabel('QAgNO3(%)')
#    axs[i].set_title('Run'+str(i+1))
#    
#    idx = np.where(plot_arr[:2304,0]==i+1)
#    data = plot_input_all.iloc[idx] #select data range
#    p =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',data = data, ax=axs[i])
#
##    plt.xlim(200,1000)
#    axs[i].set_ylim(0.5,20)
#    axs[i].set_xlim(4,45)
#fig1.colorbar(im, ax=axs[3],ticks=np.linspace(0,1,5))
#fig1.colorbar(im, ax=axs[7],ticks=np.linspace(0,1,5))   
    idx = np.where(plot_arr_1[:,0]==i+1)
    #idx1 = np.where(plot_arr_1[:,0]<i+1)
    #data_grey = plot_nn_data .iloc[idx1]
    data = plot_nn_data .iloc[idx] #select data range
    p1 =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',hue='method',data = data,s=60)
    p1.get_legend().remove()
    idx1 = np.where(plot_arr_1[:,0]<i+1)
    data1 = plot_nn_data .iloc[idx1] #select data range
    p2 =  sns.scatterplot(x='QAgNO3(%)',y='Qseed(%)',color=".5",data = data1, marker= ',')
    
    plt.ylim(0.5,20)
    plt.xlim(4,45)
    plt.title('Run'+str(i+1))
    
    fig.tight_layout()
    fig.savefig('point_meanAg0'+str(i))
#fig1.savefig('BO_manifold_Ag.pdf')

#%%
    
#%%
from PIL import Image
import glob
 
# Create the frames
frames = []
imgs = glob.glob("*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('mean_Ag.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000, loop=0)
#%%
NN_s = pd.read_excel('NN_loss.xlsx','Sheet1')
NN_s['method'] = 'NN_student'

fig = plt.figure()
BO_s = plot_input_all.iloc[:314+1051,:]

total_s = pd.concat((BO_s,NN_s))

sns.boxplot(x='Run ID',y='loss', hue='method',data= total_s)

plt.ylim(0,1)

plt.show()#X_int_5run = X_init[:1051,:]

fig.savefig('NN_student_BO.pdf')

#Y_int_5run = Y_init_1[:1051,:]
#%%
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

#from pyDOE import *
#
#exp_f = lhs(5, samples=15, criterion='center')
#
#
#bound = np.array([[4,20],[10,40],[4,20],[4,20],[200,1000]])
#
#exp_int = np.multiply(exp_f, bound[:,1]-bound[:,0])+bound[:,0]

#np.savetxt('Silver_int.csv',exp_int,delimiter=",")
#
##optimizer.plot_acquisition()
##optimizer.plot_convergence()
#
#batch_size = 10
#num_cores = 4
#
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


process_mesh1= []
#ind_mesh= []
for i in range(len(Ag)):
    for j in range(len(pva)):
        for k in range(len(tsc)):
            for m in range(len(Seed)):
                for n in range(len(tot)):
                    b = [Ag[i],pva[j],tsc[k],Seed[m],tot[n]]
#                    ind = [i,k,m,n]
            
                    process_mesh1.append(b)
        
        
process_mesh1= np.array(process_mesh1) 
process_data1,_ = batch_optimizer.model.predict(process_mesh1)


df_gp1 = pd.DataFrame(process_data, columns= [ 'QAgNO3(%)', 'Qpva(%)',
       'Qtsc(%)', 'Qseed(%)','Qtot(%)',  'loss'])
ppp = df_gp1.groupby(['QAgNO3(%)','Qseed(%)'],as_index=False).min()    
plt.contourf(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
             np.array(ppp.iloc[:,-1]).reshape(40,40),np.linspace(0,1,100),cmap='plasma')

plt.colorbar(ticks=np.linspace(0,1,5))

plt.contour(np.array(ppp.iloc[:,0]).reshape(40,40), np.array(ppp.iloc[:,1]).reshape(40,40),
         np.array(ppp.iloc[:,-1]).reshape(40,40), linestyles='dashed',colors='dimgrey')

#    plt.setp(ax.get_yticklabels(), visible=False)

plt.xlabel('QAgNO3(%)')

#    plt.xlim(200,1000)
plt.ylim(0.5,20)


batch_optimizer.model.predict()

cond = np.array(['AgNO3',	'PVA',	'NaOH',	'Hydrazine','Flow rate'
])



np.savetxt('GSNA-run_new-9.csv',batch_x_next,delimiter=",")



#%%
plt.rcParams["figure.figsize"] = [12, 8]
plot_random_bo = pd.concat((plot_input_all.iloc[:314+518,:],raw_input_3))

fig = plt.figure()
ax = sns.boxplot(x='Run ID',y='loss',hue='method',data=plot_random_bo)
plt.legend(loc= 'lowerleft')
#%%
plt.rcParams["font.family"] = "Arial"
plt.rcParams["figure.figsize"] = [6, 8]
loss3  = 1-spectra_loss_function (raw_spectra_3, target_spec_norm)

raw_input_3['loss'] = loss3
raw_input_3['method'] = 'random'
raw_input_3 = raw_input_3.sample(frac=1)
plot_random = pd.concat((plot_input_all.iloc[:314+518,:],NN_s.iloc[:30,:],raw_input_3))

plot_random1 = plot_random[plot_random['method']=='random']
plot_random1.sort_values(['Run ID', 'Condition ID'],ascending=[True, True], inplace=True)

random1 = plot_random1.values

rand_mean = plot_random1.groupby(['Run ID', 'Condition ID'])[['loss']].mean()

idx1 = []

loss1 = []
idx = []
for i in range(2):       
    idx_con = np.where(random1 [:,1]==np.argmin(rand_mean[i*15:(i+1)*15:])+1)
   

    idx1.append(idx_con[0][i*23+2:(i+1)*23])
idx = np.concatenate((idx1[0],idx1[1]))
   
fig = plt.figure()
ax = sns.stripplot(x='Run ID',y='loss',hue='method',data=plot_random1,palette="Greens")
ax = sns.boxplot(x='Run ID',y='loss',hue='method',data=plot_random1.iloc[idx,:],palette="Set2")
#ax = sns.stripplot(x='Run ID',y='loss',hue='method',data=plot_student)

ax.set_ylim([0,1])


mean_con = np.vstack(mean_con)
mini_con = np.vstack(mini_con)


 
ax.set_ylim([0,1])
fig.savefig('random.eps')


fig = plt.figure()
plot_student = plot_random[plot_random['method']=='NN_student']
fig = plt.figure()
ax = sns.stripplot(x='Run ID',y='loss',hue='method',data=plot_student,palette="Reds")


 
ax.set_ylim([0,1])
fig.savefig('student_box.eps')


#%%

loss_grid = pd.read_csv('grid_points-1207.csv', header = None)
loss_en_grid = pd.read_csv('grid_res.csv', header = None)

loss_grid['method'] = 'Grid'
loss_grid ['ensemble_loss'] = loss_en_grid
loss_bo = pd.read_csv('bo_points-1207.csv',header = None)
loss_bo['method'] = 'BO'

loss_en_bo = pd.read_csv('bo_res.csv', header = None)
loss_bo ['ensemble_loss'] = loss_en_bo
#loss_lhs = pd.read_csv('lhs_points-1207.csv',header = None)
#loss_lhs['method'] = 'LH'

loss_total = pd.concat([loss_grid, loss_bo])

loss_total.columns = [0,'1','2','3','4','loss','method','ensemble_loss']
fig = plt.figure()
ax = sns.stripplot(x='method',y='ensemble_loss',hue='method',data=loss_total )

loss_grid.to_csv('ensemble_grid.csv')
loss_bo.to_csv('ensemble_bo.csv')
ax.set_ylim([0,1])

fig.savefig('gridvsBO.eps')
