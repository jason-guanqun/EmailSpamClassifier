import scipy.io as sio
import numpy as np 
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model


data=sio.loadmat("MSdata.mat")

x_sample=data['trainx']
y_sample=data['trainy']
testx=data['testx']

# No need to sample
# idx = np.random.choice(np.arange(readx.shape[0]), readx.shape[0], replace=False)
# x_sample = readx[idx]
# y_sample = ready[idx]
# y_sample = y_sample[:,0]

# trainx=x_sample[0:(int(0.9*x_sample.shape[0]))]
# trainy=y_sample[0:(int(0.9*y_sample.shape[0]))]

# testx=x_sample[(int(0.9*x_sample.shape[0])):-1]
# testy=y_sample[(int(0.9*y_sample.shape[0])):-1]

def music_net():
	net = Sequential()

	net.add(Dense(96, input_dim=90,kernel_initializer='normal', activation='relu'))
	net.add(BatchNormalization())
	net.add(Dropout(0.5))

	net.add(Dense(128, kernel_initializer='normal', activation='relu'))
	net.add(BatchNormalization())
	net.add(Dropout(0.5))

	net.add(Dense(96, kernel_initializer='normal', activation='relu'))
	net.add(BatchNormalization())

	net.add(Dense(1, kernel_initializer='normal'))
	net.compile(loss='mean_squared_error', optimizer='adam')
	return net

model=music_net()
checkpointer = ModelCheckpoint(filepath='./best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

print "===================Training==================="
model.fit(x_sample, y_sample, epochs=100, batch_size=32, verbose=2, shuffle=True, validation_split=0.1,callbacks=[checkpointer])
model.save_weights('./last_weights.h5', overwrite=True)

print "===================Testing==================="
model = load_model('best_weights.h5')
res=model.predict(testx)
res=res[:,0]
res=np.round(res)
ans=res.astype(int)
with open('result.csv','w') as file:
	file.write('dataid,prediction'+'\n')
	for i in range(testx.shape[0]):
		file.write(str(i+1)+','+str(ans[i]))
		file.write('\n')
