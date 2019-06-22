from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import json
import wandb

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import Callback
from wandb.keras import WandbCallback
from keras.optimizers import SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3

run = wandb.init()
config = run.config
train_data_dir = '../kagfish/train'
val_data_dir = '../kagfish/train'

bb_params = ['height', 'width', 'x', 'y']
config.width = 224 # or 299
config.height = 224 # or 299
config.epochs = 25
config.batch_size = 32
config.n_train_samples=80
config.n_validation_samples=20

anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft', 'NoF']
bb_json = {}
path = "../kagfish/bbox"
for c in anno_classes:
    if c == 'other': continue # no annotation file for "other" class
    j = json.load(open('{}/{}_labels.json'.format(path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]
bb_json['img_04908.jpg']


def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (config.width / size[0])
    conv_y = (config.height / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb
  
def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plotfish(img):
    plt.imshow(to_plot(img))
    
def show_bb(i):
    bb = val_bbox[i]
    plotfish(val[i])
    plt.gca().add_patch(create_rect(bb))

    
#model=

# setup model
conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(config.width, config.height, 3))
model = Sequential()
model.add(conv_base)
model.add(Flatten(input_shape=conv_base.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
conv_base.trainable = False
optimizer = RMSprop(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (config.width, config.height),
        batch_size = config.batch_size,
        shuffle = True,
        classes = anno_classes,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(config.width, config.height),
        batch_size=config.batch_size,
        shuffle = True,
        classes = anno_classes,
        class_mode = 'categorical')

# model.fit(train_faces, train_emotions, batch_size=config.batch_size,
#         epochs=config.num_epochs, verbose=1, callbacks=[
#             WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
#         ], validation_data=(val_faces, val_emotions))


model.fit_generator(
        train_generator,
        samples_per_epoch = config.n_train_samples,
        nb_epoch = config.epochs,
        validation_data = validation_generator,
        nb_val_samples = config.n_validation_samples,
        callbacks = [best_model,WandbCallback(data_type="image", labels=anno_classes)]) #best_model,
