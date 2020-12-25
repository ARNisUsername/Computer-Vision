import numpy as np
import random
import os
from tensorflow import keras

X = []
y = []

def img_to_arr(the_path, dog_img):
    img = keras.preprocessing.image.load_img(the_path+'\\'+dog_img,
                                             target_size=(150,150))
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = img_arr / 255.0
    return img_arr

def return_images(the_path, which_one):
    X = []
    y = []
    counter = 0
    for dog_img in os.listdir(the_path):
        X.append(img_to_arr(the_path, dog_img))
        y.append(which_one)
        counter += 1
        if counter == 2000:
            break
    return (X, y)

print("Collecting dogies:")
X_dog, y_dog = return_images('C:\\Users\\arnie2014\\Desktop\\dogs', 1)
print("Collecting caties:")
X_cat, y_cat = return_images('C:\\Users\\arnie2014\\Desktop\\cats', 0)
for arr in X_cat:
    X_dog.append(arr)
for num in y_cat:
    y_dog.append(num)
X = X_dog
y = y_dog

xy = list(zip(X, y))
random.shuffle(xy)
X, y = zip(*xy)
X = np.array(list(X))
y = np.array(list(y))

model = keras.Sequential()
filters = 32
for i in range(1,4):
    filters = filters*i
    model.add(keras.layers.Conv2D(filters, kernel_size=3, activation='relu',
                                  input_shape=(150,150,3),
                                  kernel_initializer='he_uniform',
                                  padding='same'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=80, epochs=2)

model.save_weights("photo_cat_dogs_model_3")
model.load_weights("photo_cat_dogs_model_3")

def predict_animal(image_path):
    the_path = 'C:\\Users\\arnie2014\\Desktop'
    arr = img_to_arr(the_path, image_path)
    thePred = model.predict(arr.reshape(1,150,150,3))
    if thePred > 0.5:
        thePred = int(thePred*100)
        the_str = "{}% sure that this is a dog!".format(thePred)
    else:
        thePred = 100-int(thePred*100)
        the_str = "{}% sure that this is a cat!".format(thePred)
    return the_str

def test_model():
    allCats = ['catie' + str(i) + '.jpg' for i in range(1,9)]
    allDogs = ['dogie' + str(i) + '.jpg' for i in range(1,9)]
    predCats = [predict_animal(allCats[i]) for i in range(8)]
    predDogs = [predict_animal(allDogs[i]) for i in range(8)]
    correct_cats = ['cat' in predCats[i] for i in range(8)].count(True)
    correct_dogs = ['dog' in predDogs[i] for i in range(8)].count(True)
    print(f'Correct cats: {correct_cats}/8')
    print(f'Correct dogs: {correct_dogs}/8')

test_model()
