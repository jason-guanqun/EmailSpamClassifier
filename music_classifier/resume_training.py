import scipy.io as sio
import numpy as np 
import random
from keras.models import load_model

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.normalization import BatchNormalization

# from keras.optimizers import Adam

data=sio.loadmat("MSdata.mat")

readx=data['trainx']
ready=data['trainy']
testx=data['testx']

idx = np.random.choice(np.arange(readx.shape[0]), readx.shape[0], replace=False)
x_sample = readx[idx]
y_sample = ready[idx]
y_sample = y_sample[:,0]

# trainx=x_sample[0:(int(0.9*x_sample.shape[0]))]
# trainy=y_sample[0:(int(0.9*y_sample.shape[0]))]

# testx=x_sample[(int(0.9*x_sample.shape[0])):-1]
# testy=y_sample[(int(0.9*y_sample.shape[0])):-1]

model = load_model('best_weights.h5')

checkpointer = ModelCheckpoint(filepath='./best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

model.fit(x_sample, y_sample,initial_epoch=100, epochs=200, batch_size=32, verbose=2, shuffle=True, validation_split=0.1,callbacks=[checkpointer])
model.save_weights('./music_last_weights.h5', overwrite=True)
# score = model.evaluate(testx, testy, verbose=0)
# print "score=",score

# res=model.predict(testx)
# res=res[:,0]
# res=np.round(res)
# ans=res.astype(int)
# print ans[0:100]
# with open('result.csv','w') as file:
# 	file.write('dataid,prediction'+'\n')
# 	for i in range(testx.shape[0]):
# 		file.write(str(i+1)+','+str(ans[i]))
# 		file.write('\n')