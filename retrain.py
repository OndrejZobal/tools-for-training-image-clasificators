#!/usr/bin/env python3


# Begin license text.
#
# Copyright 2021 Ondřej Zobal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# End license text.


# This mutes tensorflow info logging. Not required.
import mute_tensorflow_warnings
# Used for computing class weights of the datasets.
from sklearn.utils.class_weight import compute_class_weight
# Optimizers used for training.
from tensorflow.keras.optimizers import Adam, RMSprop
# A Class used for creating a generator for the dataset.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Loading the efficient pretrained model used for transfer learning.
# from tensorflow.keras.applications.inception_v3 import \
# Layers that will be used
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
# Object for building the model
from tensorflow.keras.models import Model, Sequential
# The library for converting the model to the TensorFLow format.
from tensorflow import lite
# Importing the TensorFlow itself.
import tensorflow as tf
# A library for mathematics.
import numpy as np
# A library for interacting with the system.
import sys
# A library for multi-platforms filepaths.
import pathlib
# A library for getting the date and time (for the timestamps).
import datetime
# A library for interacting with the operating system.
import os
# For debugging
import traceback
# For exporting model information
import json

# Adjustable parameter
# Change these variables to alter the script.

# Path to the dir containing the categories.
dataset_dir = 'dataset'
# Paths to individual dataset categories (inside of 'dataset_dir').
train_dir = 'training/'
finetuning_dir = 'finetuning/'
validation_dir = 'validation/'

# TensorBoard log directory.
log_dir = 'logs/'
# Default Checkpoint folder.
checkpoint_path = 'checkpoints/'
# Default export model directory.
saved_model_path = 'saved_models/'
# Default model filename.
name = 'model'
# Selecting a training function. Can be either 'keras' or 'tf'.
training_function = 'keras'
# Name of the pretrained model to be used. Only specific models are supported.
pretrained_model_name = 'xception'

# Size of a batch.
batch_size = 8
# The default number of training cycles.
epochs_training = 10
# The default number of finetuning cycles.
epochs_finetuning = 10
# Multipliers of learning.
learning_rate_train = 1e-5
learning_rate_finetuning = 1e-7
# Prints the logo after the program loads.
print_init_banner = True

# These are changed by the script itself.
# Please don't change the values yourself.
checkpoint_name = None
# Whether skip finetuning.
skip_finetuning = False
# Whether to skip training.
skip_training = False
# Whether to not do validating
sikp_validation = False
# Whether to not validate
skip_validation = False
# Also export the trained model as lite?
save_as_lite = False

EXPERIMENTAL = False

# NONADJUSTABLE PARAMETERS
input_shape = (299, 299, 3)
# The information about the model that will be serialized into a JSON after
# the training.
info = []
# Dump of all the messages printed with the log()
log_dump = []
# The number of epochs according to the models history
initial_epoch = 1
# Keeps the value of starting epoch between training.
start_epoch = 0
# Saving the current day for info.json
date = datetime.date.today().strftime('%b-%d-%Y')
# Creating timestamp for exported files.
timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '.')
# Will be set to a path of the trained model containing the timestamp.
timestamp_path = ''
# Will be set to a list of classnames
class_names = []
# Loss functions used during training. No point in changing this.
loss = 'categorical_crossentropy'
# List of metrics the model will be tracking.
metrics = ['categorical_accuracy']
# Changing the paths into absolute ones
log_dir = pathlib.Path(log_dir)
# Used for disabling the training process itself. ex. printing help.
run = True
# A variable for models loaded through the command line arguments.
loaded_model = None
# The function that will be called on every image, before training.
preprocessing_function = None
# Is the model that will be loaded a base (headless) model or a # loaded model.
is_base = True
# List of neuron amounts of dense layers of the head of the model.
dense_amount_list = [64]
# Name of the parent dir to the model dir.
model_class_name = None
'''
# A string for
dense_amount_list_str = ''
'''
# A list of callbacks applied to the model during keras training
callbacks = []

saved_model_path = pathlib.Path(saved_model_path)


# The map translating the single char arguments into full string arguments.
def log(message, level='info', start=' ', end='\n', hide_box=False):
    global log_dump
    symbol = {
        'info': '*',
        'warning': 'W',
        'error': '!',
    }

    box = f'[{symbol.get(level)}] ' if not hide_box else ''
    nl = '\n'
    message = f'{nl if level == "error" else ""}{start}{box}{message}{end}'
    log_dump.append(message)
    print(message, end='')


def print_banner():
    print('''\
          _             _
 _ __ ___| |_ _ __ __ _(_)_ __    _ __  _   _
 | '__/ _ \ __| '__/ _` | | '_ \  | '_ \| | | |
 | | |  __/ |_| | | (_| | | | | |_| |_) | |_| |
 |_|  \___|\__|_|  \__,_|_|_| |_(_) .__/ \__, |
     by Ondřej Zobal              |_|    |___/
    ''')


# Prints the help menu.
def arg_help(next_arg):
    global run
    run = False
    log('''You can use these flags:
    -h --help\t= Print this message.
    -n --name [NAME]\t= Set the name of the model. This will appear as the \
filename of the exported model.
    -l --load [PATH]\t= Takes a relative path to a saved model.
    -b --base [PATH]\t= Takes a name of a model for transfer learning. \
It can be: xception, vgg16, mobilenet2, inception3 or resnet50.
    -B --base-custom [PATH]\t= Takes a path to a specific base model. \
    You should use --base before to set a proper preprocessing function.
    -e --epochs [AMOUNT]\t= Amount of training epochs.
    -d --dense [AMOUNT]\t= Amount of neurons in the finall dense layer.
    -c --load-checkpoint [PATH]\t= Takes a relative path to a checkpoint \
of a current model.
    -v --version\t= Displays the Tensorflow version number.
    -f  --skip-finetuning\t= When present the script will skip finetuning.
    -t  --skip-training\t= When present the script will skip training.
        --skip-validation\t= When present the script will not do validation.
        --lr-training [VALUE]\t= Sets a given float as a learning rate for \
the initial training. This is usually a very small number.
        --lr-finetuning [VALUE]\t= Sets a given float as a learning rate \
for the finetuning training. This is usually a very small number.
    -s --skip\t= Skips the training phase.''' )
    return True


# Sets the name for the model file.
def arg_name(next_arg):
    global name
    name = next_arg
    return True


# Loads the whole model (including the classification layer).
def arg_load(next_arg):
    global loaded_model, is_base, timestamp_path, info, name, \
        pretrained_model_name, training_function, dataset_dir, start_epoch,\
        initial_epoch
    path = pathlib.Path(next_arg)
    log(f'Loading a full model from {path}.')
    loaded_model = tf.keras.models.load_model(path)
    try:
        with open(pathlib.Path(next_arg).joinpath('info.json'), 'r') as file:
            info = json.load(file)
            last_info = info[0]
            name = last_info['model_name']
            pretrained_model_name = last_info['pretrained_name']
            initial_epoch = last_info['end_epoch']

            if training_function == '' or training_function is None:
                training_function = info['training_function']
            if dataset_dir == '' or dataset_dir is None:
                dataset_dir = info['dataset_dir']

            log('info.json was loaded successfully.')
    except Exception as ex:
        log('info.json was not found.', 'warning')
    is_base = False
    timestamp_path = path
    return True


# Sets the amount of epochs for training.
def arg_epochs(next_arg):
    global epochs_training, epochs_finetuning
    temp = next_arg.split(',')
    if len(temp) == 1:
        epochs_training = epochs_finetuning = int(temp[0])
    elif len(temp) == 2:
        epochs_training = int(temp[0])
        epochs_finetuning = int(temp[1])
    return True


# Sets the dense layers to be set as head of the model when building
# a new model.
def arg_dense_amount(next_arg):
    global dense_amount_list
    dense_amount_list = []
    temp = next_arg.split(',')
    for i in temp:
        dense_amount_list.append(int(i))
    return True


# Sets a checkpoint path to be loaded before training.
def arg_checkpoint(next_arg):
    global checkpoint_name
    checkpoint_name = next_arg
    return True


# Toggles skipping training.
def arg_training(next_arg):
    global skip_training
    skip_training = not skip_training
    return False


# Toggles skipping finetuning.
def arg_finetuning(next_arg):
    global skip_finetuning
    skip_finetuning = not skip_finetuning
    return False


# Toggles skipping validating
def arg_validation(next_arg):
    global skip_validation
    skip_validation = not skip_validation
    if skip_validation:
        log('Validating accuracy is disabled, this is not recomended as your '
            + 'model can start overfitting without you knowing.', 'warning')
    return False


# Sets the learning rate for training.
def arg_lr_training(next_arg):
    global learning_rate_train
    learning_rate_train = float(next_arg)
    return True


# Sets the learning rate for finetuning.
def arg_lr_finetuning(next_arg):
    global learning_rate_finetuning
    learning_rate_finetuning = float(next_arg)
    return True


# Prints TensorFlow version number.
def arg_version(next_arg):
    global run
    run = False
    log(f'You are running TensorFlow version {tf.__version__}.\n')
    return False


# Sets pretrained model to be used for training to the value of the argument.
def arg_set_pretrained_model(next_arg):
    global pretrained_model_name
    pretrained_model_name = next_arg
    return True


# Loads a specified model as the base model.
def arg_pretrained_custom(next_arg):
    global loaded_model
    path = pathlib.Path(next_arg)
    log(f'\nLoading a base model from {path}.\n')
    loaded_model = tf.keras.models.load_model(path)
    return True


# Sets it so the model will be converted to TFLite.
def arg_lite(next_arg):
    global save_as_lite
    save_as_lite = not save_as_lite
    return False


# Skips the training process.
def arg_skip(next_arg):
    global run
    run = False
    return False


# Picks training method.
def arg_pick_training_method(next_arg):
    global training_function
    if next_arg == 'tf' or next_arg == 'keras':
        training_function = next_arg
    return True


# Starts tensorboard in current log_dir. Terminates after tensorboard closes.
def arg_tensorboard(next_arg):
    os.system(f'tensorboard --logdir {log_dir}')
    exit()


def arg_default(next_arg):
    global dataset_dir
    dataset_dir = pathlib.Path(next_arg)
    return True


# Dict mapping the short forms of flags to the long ones.
char_arg_map = {
    # Short form | Long form
    'h':    'help',
    'n':    'name',
    'l':    'load',
    'b':    'base',
    'B':    'base-custom',
    'e':    'epochs',
    'c':    'load-checkpoint',
    'v':    'version',
    'd':    'dense',
    's':    'skip',
    't':    'skip-training',
    'f':    'skip-finetuning', }

# Maps the long name of a flag to a argument function.
arg_dict = {
    # Key word | Function
    'help': arg_help,
    'name': arg_name,
    'load': arg_load,
    'epochs': arg_epochs,
    'load-checkpoint': arg_checkpoint,
    'skip-finetuning': arg_finetuning,
    'skip-training': arg_training,
    'skip-validation': arg_validation,
    'lr-training': arg_lr_training,
    'lr-finetuning': arg_lr_finetuning,
    'version': arg_version,
    'dense': arg_dense_amount,
    'base-custom': arg_pretrained_custom,
    'lite': arg_lite,
    'skip': arg_skip,
    'tensorboard': arg_tensorboard,
    'base': arg_set_pretrained_model }


# Converts single char arguments into a full argument and calls
# the processing function.
def process_1char_arg(char, next_arg):
    try:
        # return process_arg(char_arg_map.get(char), next_arg)
        return arg_dict[char_arg_map[char]](next_arg)
    except Exception as e:
        log(f'Invalid single dash argument was given:\n\t{e}', 'error')


# Process command line arguments
def process_commands():
    if len(sys.argv) <= 0:
        return
    # Set to True when flag that requires additional argument after
    skip = False
    for arg in range(len(sys.argv))[1:]:
        if skip:
            skip = False
            continue
        if sys.argv[arg][0] == '-':
            next_arg = ''
            if len(sys.argv) + 1 >= arg:
                try:
                    next_arg = sys.argv[arg + 1]
                except:
                    pass
            # Handeling 'one dash per one letter' syntax.
            # This will permit passing one additional parameter
            if len(sys.argv[arg]) == 2:
                skip = process_1char_arg(sys.argv[arg][1], next_arg)
            # Long arguments
            elif len(sys.argv[arg]) > 3:
                # Handeling 'double dash, whole word! syntax.
                # This will permit passing additional parameters
                if sys.argv[arg][1] == '-':
                    skip = arg_dict[sys.argv[arg][2:]](next_arg)

        else:
            arg_default(sys.argv[arg])


def set_pretrained_model(name):
    global loaded_model
    model_dict = {
        'xception': tf.keras.applications.Xception,
        'vgg16': tf.keras.applications.VGG16,
        'mobilenet2': tf.keras.applications.MobileNetV2,
        'inception3': tf.keras.applications.InceptionV3,
        'resnet50': tf.keras.applications.ResNet50
    }

    loaded_model = model_dict[name](include_top=False, input_shape=input_shape)


def set_preprocessing_function(name):
    preprocessing_dict = {
        'xception': tf.keras.applications.xception.preprocess_input,
        'vgg16': tf.keras.applications.vgg16.preprocess_input,
        'mobilenet2': tf.keras.applications.mobilenet_v2.preprocess_input,
        'inception3': tf.keras.applications.inception_v3.preprocess_input,
        'resnet50': tf.keras.applications.resnet50.preprocess_input
    }

    try:
        return preprocessing_dict[name]
    except KeyError:
        log(f'{name} is not a supported base model.', 'error')
        exit()


# Unfreezing the model for the finetuning phase.
def unfreeze_model_loaded_model():
    global loaded_model

    loaded_model.trainable = True
    for layers in loaded_model.layers:
        layers.trainable = True


def init_training_variables():
    global loaded_model, callbacks, dataset_dir, train_dir, model_class_name,\
        validation_dir, finetuning_dir

    dataset_dir = pathlib.Path(dataset_dir)
    train_dir = dataset_dir.joinpath(train_dir)
    validation_dir = dataset_dir.joinpath(validation_dir)
    finetuning_dir = dataset_dir.joinpath(finetuning_dir)

    optimizer_training = Adam(lr=learning_rate_train)
    optimizer_finetuning = Adam(lr=learning_rate_finetuning)

    callbacks = []
    # Tensorboard setup
    # Updates files for tensorboard.
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir.joinpath(f'model_{name}_{pretrained_model_name}-e='
        + f'{initial_epoch+epochs_training+epochs_finetuning}-d='
        + f'{str(dense_amount_list).replace("[", "").replace("]", "")}-'
        + f'{timestamp}'), histogram_freq=1))

    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss',
        patience=3))

    dense_amount_list_str = str(dense_amount_list).replace("[", "")\
        .replace("]", "").replace(" ", "")

    model_class_name = f'{dataset_dir}_{pretrained_model_name}_d='\
        + f'{dense_amount_list_str}'

    # Create a category directory for the models.
    if not os.path.exists(saved_model_path.joinpath(model_class_name)):
        # Create the category directory.
        os.makedirs(saved_model_path.joinpath(model_class_name))
        # And then a checkpoint category inside.
        os.makedirs(saved_model_path.joinpath(model_class_name)
            .joinpath(checkpoint_path))

    # Save the model after every epoch.
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=saved_model_path.joinpath(model_class_name, checkpoint_path,
            f'{name}_{timestamp}_'
            + f'{initial_epoch+epochs_training+epochs_finetuning}.hdf5'),
        save_best_only=True, save_weights_only=True,
        verbose=1))

    return optimizer_training, optimizer_finetuning


# Sets class_names and num_classes and makes sure, that the amount of classes
# stays consistent.
def load_class_names(generator):
    global class_names, num_classes

    # Sets the class_names and num_classes.
    names = list(generator.class_indices.keys())
    num_classes = len(names)
    if class_names is not None:
        class_names = names
        return names

    # If an inconsistent amount of classes between dataset types appears
    # it gets reported to the user and execution is halted.
    elif len(class_names) != len(names):
        log('MISSMATCH BETWEEN THE NUMBER OF CLASSES IN A DATASET', 'error')
        log(f'Previously loaded classes: {class_names}', 'error')
        log(f'Currently loaded classes: {names}', 'error')
        exit()

    log(f'The following classes were loaded:')
    for i in range(int(len(names)/3)+1):
        try:
            log(f'{names[i*3]}', start='\t', end='')
            log(f',\t{names[i*3+1]}', end='', start='', hide_box=True)
            log(f',\t{names[i*3+2]}', start='', hide_box=True)
        except:
            break
    print()
    return names


# Initialize the generators with some values beeing passed as arguments and
# some values that it wouldn't make sense ever changeing beeing hardcoded.
def init_generators(
    preprocessing_func, directory, rotation=0, width_shift=0, height_shift=0,
    shear=0, zoom=0, h_flip=False, v_flip=False, shuffle=False):

    # Setting up an Image Data Generator.
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_func,
        rotation_range=rotation,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        shear_range=shear,
        zoom_range=zoom,
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        fill_mode='nearest')

    log('', end='')
    # Creating the actual generator form the datagen.
    generator = datagen.flow_from_directory(directory=directory,
        target_size=input_shape[0:2],
        batch_size=batch_size,
        shuffle=shuffle)

    # Calculating the steps the generator will take to loop through.
    # The training function needs to know when it has loop through the whole
    # generator, because they just loop infinitely.
    steps = generator.n / batch_size

    # Loads up the class names to the global variables.
    load_class_names(generator)

    # Calculating the imbalance in class representation of the dataset
    # so it can be accounted for in the training.
    cls_train = generator.classes
    class_weight_train = compute_class_weight(
        class_weight='balanced', classes=np.unique(cls_train), y=cls_train)


    return generator, steps, class_weight_train


# Functions that puts together the Keras model.
# It uses global values from the script.
def build_model():
    global checkpoint_name

    # Putting the model together.
    if is_base:
        set_pretrained_model(pretrained_model_name)

        # Freezing the base model for normal training.
        loaded_model.trainable = False
        for layers in loaded_model.layers:
            layers.trainable = False

        # Adding recognition layers.
        x = loaded_model.output
        x = tf.keras.layers.Flatten()(x)
        for i in dense_amount_list:
            x = Dropout(0.5)(x)
            x = Dense(i, activation='relu')(x)

        # Output layer
        out = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=loaded_model.input, outputs=out)

    else:
        model = loaded_model

    # Loading the checkpoint (if one is set).
    if checkpoint_name != None:
        if os.path.sep not in checkpoint_name:
            checkpoint_name = saved_model_path.joinpath(model_class_name,
                checkpoint_path, checkpoint_name)
        log(f'Loading a checkpoint: {checkpoint_name}.')
        model.load_weights(checkpoint_name)

    return model

# Use a custom TensorFlow function for training.
def train_with_tf(model, optimizer, generator, generator_steps,
    generator_validation, generator_validation_steps, class_weight,
    epochs, starting_epoch):
    pass
    '''
    train_acc = None
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.CategoricalCrossentropy()
    for epoch in range(epochs):
        log(f'Start of Training Epoch {epoch}')
        num_batches = 0
        for i, (x_batch, y_batch) in enumerate(generator):
            num_batches += 1
            with tf.GradientTape() as tape:
                # Running the prediction.
                y_pred = model(x_batch, training=train)
                # Calculating the loss.
                loss = loss_fn(y_batch, y_pred)

            # Calculating the gradient.
            gradients = tape.gradient(loss, model.trainable_weights)
            # Applying the gradient.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            # Updating the accuracy log.
            acc_metric.update_state(y_batch, y_pred)
            # Start a new epoch when the generator loops.
            if num_batches >= len(ds)/batch_size:
                break

        # Keeping up with the accuracy.
        train_acc = acc_metric.result()
        log(f'Accuracy over epoch {train_acc}')
        acc_metric.reset_states()

    return train_acc
    '''


# Use the build-in Keras. Fit function for training.
def train_with_keras(model, optimizer, generator, generator_steps,
    generator_validation, generator_validation_steps, class_weight, epochs,
    starting_epoch):
    accuracy = '0'

    class_weight = dict(enumerate(class_weight))

    # Compiling the model and doing the training.
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(
        x=generator,
        epochs=epochs,
        steps_per_epoch=generator_steps,
        callbacks=callbacks,
        validation_data=generator_validation,
        validation_steps=generator_validation_steps,
        class_weight=class_weight,
        initial_epoch=starting_epoch
    )
    if generator_validation is not None and\
        0 != generator_validation_steps is not None:
        accuracy = str(model.evaluate(generator_validation,
            steps=generator_validation_steps)[1] * 100)[:6]

    loss_val = history.history['loss']
    loss_val = loss_val[len(loss_val)-1]

    actual_epochs = len(history.history['loss'])

    log('', start='', hide_box=True)
    log(f'Test-set classification accuracy: {accuracy}%')
    return accuracy, loss_val, actual_epochs


# This function forwards the training based on the training function chosen in
# 'type_of_training' by the user. It can be 'tf' or 'keras'.
def train(model, optimizer, generator, generator_steps, generator_validation,
    generator_validation_steps, class_weight, epochs, starting_epoch,
    type_of_training='keras'):
    switcher = {
        'keras': train_with_keras,
        # 'tf': train_with_tf
    }

    return switcher[type_of_training](model, optimizer, generator,
        generator_steps, generator_validation, generator_validation_steps,
        class_weight, epochs, starting_epoch)


# Exports the keras model into a new directory.
def export_model(model, accuracy, loss, actual_epochs):
    global timestamp_path, info, start_epoch, initial_epoch, log_dump, name

    log('Exporting the model...')
    # Exporting the trained model.
    # Generating a unique file name.
    timestamp_path = pathlib.Path(
        f'{saved_model_path}/{model_class_name}/'
        + f'{name}_a={accuracy}%_'
        + f'e={actual_epochs}_{timestamp}')
    log(f'Exporting trained model at {timestamp_path}')
    os.mkdir(f'{timestamp_path}')
    # Save the model.
    model.save(timestamp_path)

    # Exporting the class names into the model's directory.
    try:
        names = []
        for i in range(len(class_names)):
            num = int(class_names[i].split("_")[0])
            names.append(f'{num}\n')
        with open(timestamp_path.joinpath('classes'), 'a') as file:
            for name in names:
                file.write(name)
    except:
        pass


    new_info = {
        'time': timestamp,
        'model_name': name,
        'pretrained_name': pretrained_model_name,
        'type_name': model_class_name,
        'dataset_dir': str(dataset_dir),
        'loaded_checkpoint': checkpoint_name,
        'start_epoch': initial_epoch,
        'end_epoch': actual_epochs,
        'class_number': len(class_names),
        'classes': class_names,
        'accuracy': float(accuracy),
        'loss': loss,
        'dense_layers': dense_amount_list,
        'training_function': training_function,
    }

    info.insert(0, new_info)
    # Exporting info JSON.
    with open(timestamp_path.joinpath('info.json'), 'w') as file:
        json.dump(info, file, indent=4)

    # Exporting log dump.
    # log_dump = log_dump.replace('\\n', '\n').replace('\\t', '\t')
    with open(timestamp_path.joinpath('output.txt'), 'a') as file:
        for i in log_dump:
            file.write(i)


# Exporting the keras model in the TFLite format.
def export_as_tflite():
    # Converting to the TFLite format.
    log('Saving model as TensorFlow Lite')
    converter = lite.TFLiteConverter.from_saved_model(
        str(pathlib.Path(timestamp_path)))
    tflite_model = converter.convert()
    # Preparing a filename for the new file.
    tflite_file_name = f'{timestamp_path}/{name}_{timestamp}.tflite'
    # Writing the converted IFLite model onto the drive.
    with tf.io.gfile.GFile(tflite_file_name, 'wb') as f:
        f.write(tflite_model)
    log(f'Converted model was saved to {tflite_file_name}')


# The function that gets called after the script is done initing.
def main():
    global start_epoch, initial_epoch
    # Processing standard input.
    log('All libraries loaded...')
    process_commands()

    if run:
        # Printing a cool looking banner.
        print_banner()
        optimizer_training, optimizer_finetuning = init_training_variables()

        preprocessing_function = set_preprocessing_function(
            pretrained_model_name)
        # Preparing the generators for training and validating the accuracy.
        log('Searching for datasets...')
        generator_validation = steps_validation = None
        if not skip_validation:
            generator_validation, steps_validation, cls_weight_validation = \
                init_generators(preprocessing_function, validation_dir)

        # Preparing a generator for training, if it is enabled.
        generator_train = steps_train = cls_weight_train = None
        if not skip_training:
            generator_train, steps_train, cls_weight_train = init_generators(
                preprocessing_function, train_dir, 20, 0.1, 0.1, 0.1,
                [0.9, 1.1], False, True)

        # Preparing a generator for finetuning, if it is enabled.
        generator_finetuning = steps_finetuning = cls_weight_finetuning = None
        if not skip_finetuning:
            generator_finetuning, steps_finetuning, cls_weight_finetuning = \
                init_generators(preprocessing_function, finetuning_dir, 20,
                    0.1, 0.1, 0.1, [0.9, 1.1], False, True)

        # Building the Keras model object.
        model = build_model()

        accuracy = 0
        loss = 0
        actual_epochs = 0
        # Training the model.
        try:
            # The training phase
            if not skip_training:
                log('', start='', hide_box=True)
                log('Starting the training phase.')
                next_epochs = start_epoch + epochs_training

                accuracy_train, loss_val_train, actual_epochs_train = train(
                    model,
                    optimizer_training, generator_train, steps_train,
                    generator_validation, steps_validation, cls_weight_train,
                    next_epochs, start_epoch,
                    type_of_training=training_function)

                start_epoch += epochs_training
                log('Phase training finished.')
                accuracy = accuracy_train
                loss = loss_val_train
                actual_epochs += actual_epochs_train

            # The finetuning phase
            if not skip_finetuning:
                log('', start='', hide_box=True)
                log('Starting the finetuning phase.')
                next_epochs = start_epoch + epochs_finetuning

                unfreeze_model_loaded_model()

                accuracy_finetuning, loss_val_finetuning, \
                    actual_epochs_finetuning = train(
                    model, optimizer_finetuning,
                    generator_finetuning, steps_finetuning,
                    generator_validation, steps_validation,
                    cls_weight_finetuning, next_epochs, start_epoch,
                    type_of_training=training_function)


                log('The finetuning phase finished.')
                accuracy = accuracy_finetuning
                loss = loss_val_finetuning
                actual_epochs += actual_epochs_finetuning

        except KeyboardInterrupt:
            log('Training was interrupted...', 'error')

        except:
            log('An unhandled exception caused a fatal error.', 'error')
            log(f'See', 'error', end='')
            traceback.print_exc()
            log(f'Attempting to preserve the model.', 'error')


        # Exporting the model.
        export_model(model, accuracy, loss, actual_epochs)
        if save_as_lite:
            export_as_tflite()

        step = 0
        if steps_train is not None:
            step = steps_train
        elif steps_finetuning is not None:
            step = steps_finetuning

        # Printing the summary.
        log('', start='', hide_box=True)
        log(f'TRAINING FINISHED FOR {name.upper()}')
        log(
            f'Classes: {num_classes}, epochs: {start_epoch}, '
            + f'images: {step * batch_size}, '
            + f'accuracy: {accuracy}%')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Quitting...', 'error')
