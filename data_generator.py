from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=50,
    zoom_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    vertical_flip=True)

loopsize = 1600
PATH = os.getcwd()

os.mkdir(PATH+'\\cardboard_aug')
os.mkdir(PATH+'\\paper_aug')
os.mkdir(PATH+'\\glass_aug')
os.mkdir(PATH+'\\plastic_aug')
os.mkdir(PATH+'\\metal_aug')
os.mkdir(PATH+'\\trash_aug')

os.mkdir(PATH+'\\cardboard\\cardboard_temp')
os.mkdir(PATH+'\\paper\\paper_temp')
os.mkdir(PATH+'\\glass\\glass_temp')
os.mkdir(PATH+'\\plastic\\plastic_temp')
os.mkdir(PATH+'\\metal\\metal_temp')
os.mkdir(PATH+'\\trash\\trash_temp')

# CARDBOARD
source = PATH+'\\cardboard'
dest = PATH+'\\cardboard\\cardboard_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)
image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\cardboard_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()

# GLASS
source = PATH+'\\glass'
dest = PATH+'\\glass\\glass_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)
image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\glass_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()

# METAL 
source = PATH+'\\metal'
dest = PATH+'\\metal\\metal_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)

image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\metal_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()

# PAPER

source = PATH+'\\paper'
dest = PATH+'\\paper\\paper_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)

image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\paper_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()

# PLASTIC
source = PATH+'\\plastic'
dest = PATH+'\\plastic\\plastic_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)

image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\plastic_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()

# TRASH
source = PATH+'\\trash'
dest = PATH+'\\trash\\trash_temp'
files = os.listdir(source)
for f in files:
        shutil.move(source+'\\'+f, dest)

image_generator = train_datagen.flow_from_directory(
    directory=source,
    target_size=(384,512),
    color_mode="grayscale",
    class_mode=None,
    save_to_dir=PATH+'\\trash_aug',
    batch_size=1)

for i in range(loopsize):
    image_generator.next()