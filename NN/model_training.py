#_*_ coding:utf-8
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model



#build model
model = Sequential()
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(421))
model.compile(loss='mse', optimizer='adam')


# Train the model
model.fit(
    X,
    Y,
    epochs=1000,
    batch_size=39,
    shuffle=True,
    verbose=2
)

# save well trained model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

