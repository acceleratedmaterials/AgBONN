#_*_ coding:utf-8 _*_

# prepapre data
data1=np.genfromtxt('recipe_loss_posterior8.csv',delimiter=',')
data1=np.hstack((data1[:,0].reshape(-1,1),data1[:,3].reshape(-1,1),data1[:,5].reshape(-1,1)))

df=pd.DataFrame(data1)
df=df.sort_values(by=[2])
df=df.drop_duplicates(subset=[0,1], keep='first')
data1=df.values

x=data1[:,0]
y=data1[:,1]
z=data1[:,2]

# plotting
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

np.random.seed(19680801)
npts = 200
ngridx = 200
ngridy = 200
'''
x = np.random.uniform(-2, 2, npts)
y = np.random.uniform(-2, 2, npts)
z = x * np.exp(-x**2 - y**2)
'''

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,4.1)

fig, (ax1) = plt.subplots(nrows=1)
# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
xi = np.linspace(0, 45, ngridx)
yi = np.linspace(0.5, 20, ngridy)

# Perform linear interpolation of the data (x,y)
# on a grid defined by (xi,yi)
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

# Note that scipy.interpolate provides means to interpolate data on a grid
# as well. The following would be an alternative to the four lines above:
#from scipy.interpolate import griddata
#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')


ax1.contour(xi, yi, zi,levels=14,linestyles='dashed',linewidths=0.6,colors='white')
cntr1 = ax1.contourf(xi, yi, zi, levels=np.linspace(0,1,100), cmap="plasma")

#fig.colorbar(cntr1, ax=ax1)
#fig.colorbar(cntr1, boundaries=np.linspace(0, 1, 10),ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],)
a=fig.colorbar(cntr1, boundaries=np.linspace(0, 1, 10),ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],)
a.ax.get_yaxis().labelpad = 10
a.ax.set_ylabel('Loss', rotation=90)



#ax1.plot(x, y, 'ko', ms=2,color='r')
ax1.set(xlim=(4, 45), ylim=(0.5, 20))

plt.setp(ax1.get_yticklabels(), visible=False)
#plt.setp(ax1.get_xticklabels(), visible=False)

plt.yticks([])

plt.xticks([10,20,30,40])



ax1.set(xlabel='QAgNO3(%)')

#ax1.set_title('Run 8')

plt.show()



