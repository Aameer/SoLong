{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "W&B Run: https://app.wandb.ai/univai-ss2019/kagfish/runs/28c5s5mn\n",
      "Call `%%wandb` in the cell containing your training loop to display live results.\n"
     ]
    }
   ],
   "source": [
    "from keras import backend as K\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "import json\n",
    "import wandb\n",
    "\n",
    "from keras.layers import Dense, Flatten\n",
    "from keras.models import Sequential\n",
    "from keras.callbacks import Callback\n",
    "from wandb.keras import WandbCallback\n",
    "from keras.optimizers import SGD, RMSprop\n",
    "from keras.applications.inception_v3 import InceptionV3\n",
    "\n",
    "run = wandb.init()\n",
    "config = run.config\n",
    "train_data_dir = '../kagfish/train'\n",
    "val_data_dir = '../kagfish/train'\n",
    "\n",
    "bb_params = ['height', 'width', 'x', 'y']\n",
    "config.width = 224 # or 299\n",
    "config.height = 224 # or 299\n",
    "config.epochs = 25\n",
    "config.batch_size = 32\n",
    "config.n_train_samples=80\n",
    "config.n_validation_samples=20\n",
    "\n",
    "anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft', 'NoF']\n",
    "bb_json = {}\n",
    "path = \"../kagfish/bbox\"\n",
    "for c in anno_classes:\n",
    "    if c == 'other': continue # no annotation file for \"other\" class\n",
    "    j = json.load(open('{}/{}_labels.json'.format(path, c), 'r'))\n",
    "    for l in j:\n",
    "        if 'annotations' in l.keys() and len(l['annotations'])>0:\n",
    "            bb_json[l['filename'].split('/')[-1]] = sorted(\n",
    "                l['annotations'], key=lambda x: x['height']*x['width'])[-1]\n",
    "bb_json['img_04908.jpg']\n",
    "\n",
    "\n",
    "def convert_bb(bb, size):\n",
    "    bb = [bb[p] for p in bb_params]\n",
    "    conv_x = (config.width / size[0])\n",
    "    conv_y = (config.height / size[1])\n",
    "    bb[0] = bb[0]*conv_y\n",
    "    bb[1] = bb[1]*conv_x\n",
    "    bb[2] = max(bb[2]*conv_x, 0)\n",
    "    bb[3] = max(bb[3]*conv_y, 0)\n",
    "    return bb\n",
    "  \n",
    "def create_rect(bb, color='red'):\n",
    "    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)\n",
    "\n",
    "def to_plot(img):\n",
    "    if K.image_dim_ordering() == 'tf':\n",
    "        return np.rollaxis(img, 0, 1).astype(np.uint8)\n",
    "    else:\n",
    "        return np.rollaxis(img, 0, 3).astype(np.uint8)\n",
    "\n",
    "def plotfish(img):\n",
    "    plt.imshow(to_plot(img))\n",
    "    \n",
    "def show_bb(i):\n",
    "    bb = val_bbox[i]\n",
    "    plotfish(val[i])\n",
    "    plt.gca().add_patch(create_rect(bb))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft', 'NoF']\n"
     ]
    }
   ],
   "source": [
    "print(anno_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 3708 images belonging to 8 classes.\n",
      "Found 3708 images belonging to 8 classes.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:61: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.\n",
      "/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:61: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=<keras_pre..., callbacks=[<wandb.ke..., steps_per_epoch=2, epochs=25, validation_steps=20)`\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/25\n"
     ]
    },
    {
     "ename": "OSError",
     "evalue": "cannot identify image file '../kagfish/train/yft/img_06687.jpg'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mOSError\u001b[0m                                   Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-6-47e68f923564>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     59\u001b[0m         \u001b[0mvalidation_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalidation_generator\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     60\u001b[0m         \u001b[0mnb_val_samples\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconfig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mn_validation_samples\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 61\u001b[0;31m         callbacks = [WandbCallback()]) #best_model,\n\u001b[0m",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/legacy/interfaces.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     89\u001b[0m                 warnings.warn('Update your `' + object_name + '` call to the ' +\n\u001b[1;32m     90\u001b[0m                               'Keras 2 API: ' + signature, stacklevel=2)\n\u001b[0;32m---> 91\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     92\u001b[0m         \u001b[0mwrapper\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_original_function\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     93\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/engine/training.py\u001b[0m in \u001b[0;36mfit_generator\u001b[0;34m(self, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)\u001b[0m\n\u001b[1;32m   1416\u001b[0m             \u001b[0muse_multiprocessing\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0muse_multiprocessing\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1417\u001b[0m             \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mshuffle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1418\u001b[0;31m             initial_epoch=initial_epoch)\n\u001b[0m\u001b[1;32m   1419\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1420\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0minterfaces\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlegacy_generator_methods_support\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/engine/training_generator.py\u001b[0m in \u001b[0;36mfit_generator\u001b[0;34m(model, generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps, class_weight, max_queue_size, workers, use_multiprocessing, shuffle, initial_epoch)\u001b[0m\n\u001b[1;32m    179\u001b[0m             \u001b[0mbatch_index\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    180\u001b[0m             \u001b[0;32mwhile\u001b[0m \u001b[0msteps_done\u001b[0m \u001b[0;34m<\u001b[0m \u001b[0msteps_per_epoch\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 181\u001b[0;31m                 \u001b[0mgenerator_output\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_generator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    182\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    183\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgenerator_output\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'__len__'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/utils/data_utils.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    707\u001b[0m                     \u001b[0;34m\"`use_multiprocessing=False, workers > 1`.\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    708\u001b[0m                     \"For more information see issue #1638.\")\n\u001b[0;32m--> 709\u001b[0;31m             \u001b[0msix\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreraise\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexc_info\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/six.py\u001b[0m in \u001b[0;36mreraise\u001b[0;34m(tp, value, tb)\u001b[0m\n\u001b[1;32m    691\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    692\u001b[0m                 \u001b[0;32mraise\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 693\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    694\u001b[0m         \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    695\u001b[0m             \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/utils/data_utils.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    683\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    684\u001b[0m             \u001b[0;32mwhile\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mis_running\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 685\u001b[0;31m                 \u001b[0minputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mqueue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mblock\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    686\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mqueue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtask_done\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    687\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0minputs\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/multiprocessing/pool.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    655\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_value\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    656\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 657\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_value\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    658\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    659\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_set\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/multiprocessing/pool.py\u001b[0m in \u001b[0;36mworker\u001b[0;34m(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)\u001b[0m\n\u001b[1;32m    119\u001b[0m         \u001b[0mjob\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtask\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    120\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 121\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    122\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    123\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mwrap_exception\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mfunc\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0m_helper_reraises_exception\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras/utils/data_utils.py\u001b[0m in \u001b[0;36mnext_sample\u001b[0;34m(uid)\u001b[0m\n\u001b[1;32m    624\u001b[0m         \u001b[0mThe\u001b[0m \u001b[0mnext\u001b[0m \u001b[0mvalue\u001b[0m \u001b[0mof\u001b[0m \u001b[0mgenerator\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m`\u001b[0m\u001b[0muid\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    625\u001b[0m     \"\"\"\n\u001b[0;32m--> 626\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0msix\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_SHARED_SEQUENCES\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0muid\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    627\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    628\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras_preprocessing/image/iterator.py\u001b[0m in \u001b[0;36m__next__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m     98\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     99\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__next__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 100\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    101\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    102\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras_preprocessing/image/iterator.py\u001b[0m in \u001b[0;36mnext\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    110\u001b[0m         \u001b[0;31m# The transformation of images is not under thread lock\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    111\u001b[0m         \u001b[0;31m# so it can be done in parallel\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 112\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_batches_of_transformed_samples\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindex_array\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    113\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    114\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_get_batches_of_transformed_samples\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindex_array\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras_preprocessing/image/iterator.py\u001b[0m in \u001b[0;36m_get_batches_of_transformed_samples\u001b[0;34m(self, index_array)\u001b[0m\n\u001b[1;32m    224\u001b[0m                            \u001b[0mcolor_mode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolor_mode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    225\u001b[0m                            \u001b[0mtarget_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtarget_size\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 226\u001b[0;31m                            interpolation=self.interpolation)\n\u001b[0m\u001b[1;32m    227\u001b[0m             \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mimg_to_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimg\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata_format\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_format\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    228\u001b[0m             \u001b[0;31m# Pillow images should be closed after `load_img`,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/keras_preprocessing/image/utils.py\u001b[0m in \u001b[0;36mload_img\u001b[0;34m(path, grayscale, color_mode, target_size, interpolation)\u001b[0m\n\u001b[1;32m    102\u001b[0m         raise ImportError('Could not import PIL.Image. '\n\u001b[1;32m    103\u001b[0m                           'The use of `array_to_img` requires PIL.')\n\u001b[0;32m--> 104\u001b[0;31m     \u001b[0mimg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpil_image\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    105\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mcolor_mode\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'grayscale'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    106\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mimg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m'L'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.7/site-packages/PIL/Image.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(fp, mode)\u001b[0m\n\u001b[1;32m   2703\u001b[0m         \u001b[0mwarnings\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwarn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmessage\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2704\u001b[0m     raise IOError(\"cannot identify image file %r\"\n\u001b[0;32m-> 2705\u001b[0;31m                   % (filename if filename else fp))\n\u001b[0m\u001b[1;32m   2706\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2707\u001b[0m \u001b[0;31m#\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mOSError\u001b[0m: cannot identify image file '../kagfish/train/yft/img_06687.jpg'"
     ]
    }
   ],
   "source": [
    "\n",
    "#model=\n",
    "\n",
    "# setup model\n",
    "conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(config.width, config.height, 3))\n",
    "model = Sequential()\n",
    "model.add(conv_base)\n",
    "model.add(Flatten(input_shape=conv_base.output_shape[1:]))\n",
    "model.add(Dense(256, activation='relu'))\n",
    "model.add(Dense(8, activation='sigmoid'))\n",
    "conv_base.trainable = False\n",
    "optimizer = RMSprop(lr=1e-5)\n",
    "model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])\n",
    "\n",
    "# autosave best Model\n",
    "best_model_file = \"./weights.h5\"\n",
    "best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)\n",
    "\n",
    "train_datagen = ImageDataGenerator(\n",
    "        rescale=1./255,\n",
    "        shear_range=0.1,\n",
    "        zoom_range=0.1,\n",
    "        rotation_range=10.,\n",
    "        width_shift_range=0.1,\n",
    "        height_shift_range=0.1,\n",
    "        horizontal_flip=True)\n",
    "        #vlidation_split=0.2)\n",
    "\n",
    "# this is the augmentation configuration we will use for validation:\n",
    "# only rescaling\n",
    "val_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "        train_data_dir,\n",
    "        target_size = (config.width, config.height),\n",
    "        batch_size = config.batch_size,\n",
    "        shuffle = True,\n",
    "        classes = anno_classes,\n",
    "        class_mode = 'categorical')\n",
    "\n",
    "validation_generator = val_datagen.flow_from_directory(\n",
    "        val_data_dir,\n",
    "        target_size=(config.width, config.height),\n",
    "        batch_size=config.batch_size,\n",
    "        shuffle = True,\n",
    "        classes = anno_classes,\n",
    "        class_mode = 'categorical')\n",
    "\n",
    "# model.fit(train_faces, train_emotions, batch_size=config.batch_size,\n",
    "#         epochs=config.num_epochs, verbose=1, callbacks=[\n",
    "#             WandbCallback(data_type=\"image\", labels=[\"Angry\", \"Disgust\", \"Fear\", \"Happy\", \"Sad\", \"Surprise\", \"Neutral\"])\n",
    "#         ], validation_data=(val_faces, val_emotions))\n",
    "\n",
    "\n",
    "model.fit_generator(\n",
    "        train_generator,\n",
    "        samples_per_epoch = config.n_train_samples,\n",
    "        nb_epoch = config.epochs,\n",
    "        validation_data = validation_generator,\n",
    "        nb_val_samples = config.n_validation_samples,\n",
    "        callbacks = [WandbCallback()]) #best_model,\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
