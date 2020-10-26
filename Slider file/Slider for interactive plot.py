#-*-conding:utf-8-*-

# import libraries for interactive plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# load the well trained model
model = load_model('model.h5')


# data point for x-coordinate
wavelength=list(range(380,801,1))



# set up the figure
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)

t = np.arange(1.0, 4.0, 1)
s = np.arange(1.0, 4.0, 1)
l, = plt.plot(t, s, lw=2)

plt.ylim((0, 1.2))
plt.xlim((380,800))

ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'

# define the position of the sliding bar of the 5 variables
ax_ag = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_pva = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_tsc = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_seed = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_tot = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor=axcolor)

# define the range of the 5 variables
Qag = Slider(ax_ag, 'ag', 0.5, 80, valinit=0.13, valstep=0.001)
Qpva = Slider(ax_pva, 'pva', 10, 40, valinit=0.33, valstep=0.001)
Qtsc = Slider(ax_tsc, 'tsc', 0.5, 80, valinit=0.056, valstep=0.001)
Qseed = Slider(ax_seed, 'seed', 0.5, 80, valinit=0.045, valstep=0.001)
Qtot = Slider(ax_tot, 'tot', 200, 1000, valinit=227, valstep=1)


#
def update(val):
    ag = Qag.val
    pva=Qpva.val
    tsc=Qtsc.val
    seed=Qseed.val
    tot=Qtot.val
    
    input=np.array([[float(ag),float(pva),float(tsc),float(seed),float(tot)]])
    data=model.predict(input)

    l.set_ydata(data)
    l.set_xdata(wavelength)
    fig.canvas.draw_idle()

Qag.on_changed(update)
Qpva.on_changed(update)
Qtsc.on_changed(update)
Qseed.on_changed(update)
Qtot.on_changed(update)


# reset button
resetax = plt.axes([0.8, 0.8, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    Qag.reset()
button.on_clicked(reset)

plt.show()
