# USAGE
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
# PATHS.LIST_IMAGES agarra todos los PATH a todos los input images de mi --dataset directory y hace una lista de PATHs
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images PATH
for imagePath in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring
    # aspect ratio), flatten the image into 32x32x3=3072 pixel image		#nota: tenemos 1000 imagenes de 3 categorias. Cada una queda de 32x32, cada categoria
    # into a list, and store the image in the data list						#queda de 1024 * 3 categorias = 3072. O sea, que nuestra primera layer de la NN va a tener 3072 nodos.
    image = cv2.imread(imagePath)  # loadea en ram la memoria pasandole el PATH de la imagen en el --dataset
    image = cv2.resize(image, (
        32, 32)).flatten()  # cv2.resize(src, (size)) #array.flatten() de numpy devuelve el array en 1 dimension.
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1] from [0,255]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)  # pasamos la lista de labels de imagenes a un numpy array

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for
# testing USANDO LA API SCIKIT_LEARN //// trainX y testX es para la image data y trainY y testY para los labels
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# Hasta ahora el label de cada imagen se representa como una string que la saque del PATH.
# Pero Keras necesita primero pasar el label a Integer y luego a un vector de bits, usando one-hot encoding.
# ver>https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/ para esto que acabo de explicar.

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
# queda asi>> [1, 0, 0] # corresponds to cats
#			 [0, 1, 0] # corresponds to dogs
#			 [0, 0, 1] # corresponds to panda
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras. Nuestra nn va a tener 1 input layer de 3072 nodos,
# 2 hidden layers y 1 output layer de 3 nodos para cada una de de las 3 categorias, perros, gatos y pandas.
# ACA DEFINIMOS EL MODELO DE LA NN
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 80

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the neural network
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY),
              epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"], save_format="h5")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
