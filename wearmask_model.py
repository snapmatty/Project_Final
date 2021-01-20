from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

imageset_path = list(paths.list_images('data/dataset'))
dataset = []
labels = []

# looping image pathway
for imagePath in imageset_path:
    # extracting the label of class from the name of file
    label = imagePath.split(os.path.sep)[-2]
    # loading the image with set size (in target_size) and preprocessing
    img = load_img(imagePath, target_size=(224, 224))
    # converting the image to a numpy array
    img = img_to_array(img)
    # preprocessesing a numpy array encoding a batch of images
    img = preprocess_input(img)
    # updating dataset and labels lists
    dataset.append(img)
    labels.append(label)
# converting the dataset and labels to numpy arrays
dataset = np.array(dataset, dtype="float32")
labels = np.array(labels)

mod_base = MobileNetV2(weights="imagenet", include_top=False,input_shape=(224, 224, 3))
# construct the head of the model that will be placed on top of the base model
mod_head = mod_base.output
mod_head = AveragePooling2D(pool_size=(7, 7))(mod_head)
mod_head = Flatten(name="flatten")(mod_head)
mod_head = Dense(128, activation="relu")(mod_head)
mod_head = Dropout(0.5)(mod_head)
mod_head = Dense(2, activation="softmax")(mod_head)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=mod_base.input, outputs=mod_head)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in mod_base.layers:
    layer.trainable = False

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(dataset, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# saving the trained model
model.save('wearmask_model_ver1.h5')
