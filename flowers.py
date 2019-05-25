import numpy as np
import scipy
from scipy import ndimage
import h5py
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout

classes=["phlox","rose","calendula","iris","leucanthemum maximum","bellflower","viola","rudbeckia laciniata","peony","aquilegia"]

#geting the data from h5 file
with h5py.File('./Data/FlowerColorImages.h5', 'r') as f:
    flower_tensors = f['images'].value
    flower_targets = f['labels'].value
#print(flower_tensors)
#print(flower_targets)

#spliting the data fot test and train
x_train, x_test, y_train, y_test = train_test_split(flower_tensors, flower_targets, test_size = 0.2, random_state = 1)
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#spliting the test data to test and validate
n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]
#print(x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape)

#normalize the input
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_valid = x_valid.astype('float32')/255

#converts the classes into binary matrix
c_y_train = to_categorical(y_train, 10)
c_y_test = to_categorical(y_test, 10)
c_y_valid = to_categorical(y_valid, 10)
#print(x_train.shape, c_y_train.shape, x_test.shape, c_y_test.shape, x_valid.shape, c_y_valid.shape)

#creating the model
def mlp_model():
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(128*128*3,)))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#training
mlp_model = mlp_model()
mlp_history = mlp_model.fit(x_train.reshape(-1, 128*128*3), c_y_train,
                                  validation_data=(x_valid.reshape(-1, 128*128*3), c_y_valid),
                                  epochs=50, batch_size=64, verbose=2)


#testing after training
mlp_test_score = mlp_model.evaluate(x_test.reshape(-1, 128*128*3), c_y_test)
print("test loss: ",mlp_test_score[0],"\ntest acc: ",mlp_test_score[1],"\n")



#print the predicted flowers from test
"""
predictions=mlp_model.predict(x_test.reshape(-1, 128*128*3))
for i in range (0,21):
    max=predictions[i][0]
    index=0
    for j in range(1,10):
        if (predictions[i][j]>max):
            max=predictions[i][j]
            index=j
    for j in range (0,10):
        predictions[i][j]=0

    print("predicted flower",i,"=",classes[index],"\n actual flower",i,"=",classes[y_test[i]],"\n")
"""


#test your own flower
"""
my_image = "rose1.jpg"  #image name here
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(128,128)).reshape((1, 128*128*3))
predicted=mlp_model.predict(my_image)
max=predicted[0][0]
index=0
for i in range (0,10):
    if (predicted[0][i]>max):
            max=predicted[0][i]
            index=i
print(predicted[0])
print("predicted flower is: ",classes[index])

"""
