import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# USE GPU IF AVAILABLE
try:
    import tensorflow as tf
    if tf.test.is_gpu_available():
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)
except:
        pass


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


def accuracy(cm):
    return ((cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0]+ cm[0][0] + cm[1][1]))

    
#...................* SELF MADE NEURAL NETWORK *........................#
import self_made_neural as nn
layer_size = [11,6,6,1]
net = nn.neural_network(layer_size)
net.driver(X_train, y_train, epochs = 100, batch_size = 10)
y_pred = net.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy(cm)
#........................................................................#

#........................* KERAS NEURAL NETWORK *........................#
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred_keras = classifier.predict(X_test)
y_pred_keras = (y_pred_keras > 0.5)
cm_2 = confusion_matrix(y_test, y_pred_keras)
acc_2 = accuracy(cm_2)
#.........................* PRINTING THE RESULTS. *...................................#
print(f"Accuracy of Self-made Net: {acc*100} %")
print(f"Accuracy of keras Net: {acc_2*100}%")

