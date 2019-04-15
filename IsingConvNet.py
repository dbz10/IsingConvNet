import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical



base_dir = '/Users/danielben-zion/Projects/IsingConvNet'


train_dir = os.path.join(base_dir,'train_imgs')
train_para_dir = os.path.join(train_dir,'paramagnet')
train_crit_dir = os.path.join(train_dir,'critical')
train_ordered_dir = os.path.join(train_dir,'ordered')

print('Total training paramagnetic images: ', len(os.listdir(train_para_dir)))
print('Total training critical images: ', len(os.listdir(train_crit_dir)))
print('Total training ordered images: ', len(os.listdir(train_ordered_dir)))


val_dir = train_dir = os.path.join(base_dir,'val_imgs')
# os.mkdir(val_dir)

val_para_dir = os.path.join(val_dir,'paramagnet')
val_crit_dir = os.path.join(val_dir,'critical')
val_ordered_dir = os.path.join(val_dir,'ordered')

# os.mkdir(val_para_dir)
# os.mkdir(val_crit_dir)
# os.mkdir(val_ordered_dir)

train_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip = True,
	samplewise_center=True)

val_datagen = ImageDataGenerator(
	horizontal_flip = True,
	vertical_flip = True,
	samplewise_center=True)

img_size = 64

# downsampling images seems to be important 
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(img_size,img_size),
	batch_size=60,
	class_mode='categorical',
	color_mode = 'grayscale')

val_generator = train_datagen.flow_from_directory(
	val_dir,
	target_size=(img_size,img_size),
	batch_size=60,
	class_mode='categorical',
	color_mode = 'grayscale')

model = models.Sequential()
model.add(layers.Conv2D(6, (3,3), activation='relu', input_shape = (img_size,img_size,1)))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(4, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(2, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
	optimizer = 'RMSprop',
	metrics=[metrics.categorical_accuracy],
	)

# checking that images are properly formatted
# for data_batch,labels_batch in train_generator:
# 	print('data batch shape: ',data_batch.shape)
# 	print('labels batch shape', labels_batch.shape)
# 	break

history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data = val_generator,
	validation_steps = 50)

model.save('ising_snapshots_small.h5')

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc,'r-',label='Validation Accuracy')

plt.legend()
plt.show()
