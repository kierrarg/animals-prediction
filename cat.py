from matplotlib import pyplot
from matplotlib.image import imread
import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator



folder = "data/train/cat/"
folder0 = "data/train/ferret/"
folder3 = "data/train/dog/"
folder1 = "data/test/cat/"
folder2 = "data/test/ferret/"
folder4 = "data/test/dog/"

#displaying table of cats used for training

#resizing the images would be optimal but I am using a smaller training set and have decided against it

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    file = folder + 'cat.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    file = folder0 + 'ferret.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    file = folder3 + 'dog.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

#developing baseline CNN using ReLU
def defineModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape = (200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
    #dropout regularization
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    #compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#learning curves
def summarizeDiagnostics(history):
     # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

#test evaluating models
def testHarness():
    # define model
    model = defineModel()
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow_from_directory('data/train/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory('data/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
    validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarizeDiagnostics(history)

for i in range(4):
    pyplot.subplot(330 + 1 + i)
    file = folder1 + 'cat.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

for i in range(4):
    pyplot.subplot(330 + 1 + i)
    file = folder2 + 'ferret.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

for i in range(4):
    pyplot.subplot(330 + 1 + i)
    file = folder4 + 'dog.' + str(i) + '.png'
    img = imread(file)
    pyplot.imshow(img)

pyplot.show()

testHarness()
