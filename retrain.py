#!/usr/bin/python3

'''
Begin license text.

Copyright 2020 Ondřej Zobal

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

End license text.
'''

# This mutes tensorflow info logging. Not required.
import mute_tensorflow_warnings
# Used for computing class weights of the datasets.
from sklearn.utils.class_weight import compute_class_weight
# Oprimizers used for training.
from tensorflow.keras.optimizers import Adam, RMSprop
# A Class used for creating a generator for the dataset.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Loading the efficient pretrained model used for transfer learning.
from tensorflow.keras.applications.inception_v3 import \
    preprocess_input as PreProcess  # TODO something about choosing other models.
# Layers that will be used
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
# Object for building the model
from tensorflow.keras.models import Model, Sequential
# The library for converting the model to the TensorFLow format.
from tensorflow import lite
# Importing the TensorFlow itself.
import tensorflow as tf
# A library for drawing graphs.
import matplotlib.pyplot as plt
# A library for manipulating images.
import PIL
# A library for mathematics.
import numpy as np
# A library for interacting with the system.
import sys
# A library for multy-platforms filepaths.
import pathlib
# A library for getting the date and time (for the timestamps).
import datetime
# A library for interacting with the operating system.
import os

'''  PARAMETERS - Change these values. '''
# Paths to individual dataset categories
train_dir = 'Training/'
validation_dir = 'Validation/'
finetuning_dir = 'Finetuning/'
ds_dir = 'dataset/'

log_dir = 'logs/'  # TensorBoard log directory
checkpoint_path = 'checkpoints/'  # Default Checkpoint folder
saved_model_path = 'saved_models/'  # Default export model directory
name = 'model'  # Default model filename
model_class_name = 'modelclass' # Name of the parent dir to the model dir

loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
batch_size = 8  # Size of a batch
epochs = 20  # The default number of training cycles
dense_amount = 64  # The size of the classification dense layer
dense_count = 1
learning_rate_train = 1e-5
learning_rate_finetuning = 1e-7

# These are changed by the script itself. Please don't change the values yourself
print_init_banner = True
checkpoint_name = None
skip_finetuning = False
skip_training = False
run = True  # Used for disabling the training process itself. ex. printing help
# A variable for models loaded through the command line arguments.
loaded_model = None
is_base = True
save_as_lite = False  # Also export the trained model as lite?
timestamp = str(datetime.datetime.now()).replace(
    " ", "-").replace(":", ".")  # Creating timestamp for exported  files
timestamp_path = ''
class_names = None

PreTrainedModel = tf.keras.applications.InceptionV3(
    include_top=False, input_shape=(299, 299, 3))

# Changing the paths into absolute ones.
log_dir = pathlib.Path(log_dir)
checkpoint_path = pathlib.Path(checkpoint_path)
saved_model_path = pathlib.Path(saved_model_path)
ds_dir = pathlib.Path(ds_dir)
train_dir = ds_dir.joinpath(train_dir)
validation_dir = ds_dir.joinpath(validation_dir)
finetuning_dir = ds_dir.joinpath(finetuning_dir)

EXPERIMENTAL = False

'''  PROCESSING ARGUMENTS  '''
# The map translating the single char arguments into full string arguments.

def log(message, level = 'info'):
    symbol = {
        'info': '*',
        'varning': '@',
        'error': '!',
    }

    print(f' [{symbol.get(level)}] {message}')


'''
def train(model, train, ds, epoch, lr, sample_weight):
    train_acc = None
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.CategoricalCrossentropy()
    for epoch in range(epochs):
        log(f'Start of Training Epoch {epoch}')
        num_batches = 0
        for i, (x_batch, y_batch) in enumerate(ds):
            num_batches += 1
            with tf.GradientTape() as tape:
                # Running the prediction
                y_pred = model(x_batch, training=train)
                # Calculating the loss
                loss = loss_fn(y_batch, y_pred)

            # Calculating the gradiant
            gradients = tape.gradient(loss, model.trainable_weights)
            # Applying the gradient
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            # Updating the accuracy log
            acc_metric.update_state(y_batch, y_pred)
            # Start a new epoch when the generator loops
            if num_batches >= len(ds)/batch_size:
                break

        # Keeping up with the accuracy
        train_acc = acc_metric.result()
        log(f'Accuracy over epoch {train_acc}')
        acc_metric.reset_states()

    return train_acc
'''

# Prints the help menu
def arg_help(next_arg):
    global run
    run = False
    log(
        '''\nYou can use these flags:\n
        -h --help\t= Print this message.\n
        -n --name\t= Set the name of the model. This will appear as the filename of the exported model.\n
        -l --load [PATH]\t= Takes a relative path to a saved model.\n
        -b --base [PATH]\t= Takes a relative path to a base model for tranfer-learning.\n
        -e --epochs [AMOUNT]\t= Amount of training epochs.\n
        -d --dense [AMOUNT]\t= Amount of neurons in the finall dense layer.\n
        -c --count [AMOUNT]\t= The amount of dense layers at the end.\n
        -C --load-checkpoint [PATH]\t= Takes a relative path to a checkpoint of a current model.\n
        -v --version\t= Displays the Tensorflow version number.\n
           --skip-finetuning\t= When present the script will skip finetuning.\n
           --lr-training [VALUE]\t= Sets a given float as a learning rate for the initial training. This is ussually a very small number.\n
           --lr-finetuning [VALUE]\t= Sets a given float as a learning rate for the finetuning training. This is ussually a very small number.\n
        -s --skip\t= Skips the training phase.\n
        -t --tensorboard\t= Runs tensor board on current logdir.\n''')
    return True


# Sets the name for the model file
def arg_name(next_arg):
    global name
    name = next_arg
    return True


# Loads the whole model (including the classification layer)
def arg_load(next_arg):
    global loaded_model
    global is_base
    global timestamp_path
    path = pathlib.Path(next_arg)
    log(
        f'\nLoading a full model from {path}.\n')
    loaded_model = tf.keras.models.load_model(path)
    is_base = False
    timestamp_path = path
    return True


# Sets the amount of epochs
def arg_epochs(next_arg):
    global epochs
    epochs = int(next_arg)
    return True


# Sets a checkpoint path to be loaded before training
def arg_checkpoint(next_arg):
    global checkpoint_name
    checkpoint_name = next_arg
    return True


# Sets the finetuning to be skipped
def arg_finetuning(next_arg):
    global skip_finetuning
    skip_finetuning = True
    return False


# Sets the learning rate for training
def arg_lr_training(next_arg):
    global learning_rate_train
    learning_rate_train = float(next_arg)
    return True


# Sets the learning rate for finetuning
def arg_lr_finetuning(next_arg):
    global learning_rate_finetuning
    learning_rate_finetuning = float(next_arg)
    return True


# Prints TensorFlow version number
def arg_version(next_arg):
    global run
    run = False
    log(
        f'\nThis script was made for Tensorflow 2.2. You are running version {tf.__version__}.\n')
    return False


# Sets the classification dense layer size to specified value
def arg_dense(next_arg):
    global dense_amount
    dense_amount = int(next_arg)
    return True


# The count of dense layers at the recognition layer
def arg_dense_count(next_arg):
    global dense_count
    dense_count = int(next_arg)
    return True


# Loads a specified model as the base model
def arg_base(next_arg):
    global loaded_model
    path = pathlib.Path(next_arg)
    log(f'\nLoading a base model from {path}.\n')
    loaded_model = tf.keras.models.load_model(path)
    return True


# Sets it so the model will be converted to TFLite
def arg_lite(next_arg):
    global save_as_lite
    save_as_lite = not save_as_lite
    return False


# Skips the training process
def arg_skip(next_arg):
    global run
    run = False
    return False


# Starts tensorboard in current log_dir. Terminates after tensorboard closes
def arg_tensorboard(next_arg):
    os.system(f'tensorboard --logdir {log_dir}')
    exit()


char_arg_map = {
    'h': 'help',
    'n': 'name',
    'l': 'load',
    'b': 'base',
    'e': 'epochs',
    'C': 'load-checkpoint',
    'v': 'version',
    'd': 'dense',
    'c': 'count',
    's': 'skip',
    't': 'tensorboard'}

arg_dict = {
    # Key   Function    Short form
    'help': arg_help,
    'name': arg_name,
    'load': arg_load,
    'epochs': arg_epochs,
    'load-checkpoint': arg_checkpoint,
    'skip-finetuning': arg_finetuning,
    'lr-training': arg_lr_training,
    'lr-finetuning': arg_lr_finetuning,
    'version': arg_version,
    'dense': arg_dense,
    'count': arg_dense_count,
    'base': arg_base,
    'lite': arg_lite,
    'skip': arg_skip,
    'tensorboard': arg_tensorboard,
}


# Return a list of filenames attached to given path
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


# Converts single char arguments into a full argument and calls the processing function.
def process_1char_arg(char, next_arg):
    try:
        # return process_arg(char_arg_map.get(char), next_arg)
        return arg_dict[char_arg_map[char]](next_arg)
    except Exception as e:
        log(f'\nInvalid single dash argument was given:\n\t{e}', 'error')


# Process command line arguments args
def process_commands():
    if len(sys.argv) > 0:
        skip = False  # Set to True when flag that requires aditional argument after
        for arg in range(len(sys.argv)):
            skip = False
            if not skip:
                if (sys.argv[arg][0] == '-'):
                    next_arg = ''
                    if len(sys.argv) + 1 >= arg:
                        try:
                            next_arg = sys.argv[arg + 1]
                        except:
                            None
                    # Single letter arguments
                    if len(sys.argv[
                            arg]) == 2:  # Handeling 'one dash per onle letter' syntax. This will permit passing one aditional argument
                        skip = process_1char_arg(sys.argv[arg][1], next_arg)
                    # Word arguments
                    elif len(sys.argv[arg]) > 3:
                        if sys.argv[arg][
                                1] == '-':  # Handeling 'double dash, whole word! syntax. This will permit passing aditional arguments
                            # skip = process_arg(sys.argv[arg][2:], next_arg)
                            skip = arg_dict[sys.argv[arg][2:]](next_arg)


'''  SETTING UP CALLBACKS, GENERATORS, ETC...  '''
def init_training_variables():
    global loaded_model, input_shape, optimizer, optimizer_finetuning, tensorboard_callback, model_class_name, checkpoint_callback

    # Load the default starting model
    if loaded_model == None:
        loaded_model = PreTrainedModel  # TODO Here you can change the default model

    input_shape = loaded_model.input.shape[1:3]

    # Here you can change the optimizers
    # TODO Here you can change the default optimizer
    optimizer = Adam(lr=learning_rate_train)
    optimizer_finetuning = Adam(
        lr=learning_rate_finetuning)  # TODO Here you can change the default optimizer for finetuning

    # Tensorboard setup
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir.joinpath(f'model_{name}-e={epochs}-d={dense_amount}-\
        {timestamp}'), histogram_freq=1)  # Updates files for tensorboard

    model_class_name += f'_d{dense_amount}-c{dense_count}'

    # Create a category directory for the models
    if not os.path.exists(saved_model_path.joinpath(model_class_name)):
        # Create the category directory
        os.makedirs(saved_model_path.joinpath(model_class_name))
        # And then a checkpoint category insede
        os.makedirs(saved_model_path.joinpath(model_class_name).joinpath('Checkpoints'))

    # Save the model after every epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_path.joinpath(model_class_name)
                                                            .joinpath(f'{name}_{timestamp}_{epochs:02d}.hdf5'),
                                                            save_best_only=True, save_weights_only=True,
                                                            verbose=1)

   
def load_class_names(generator):
    global class_names, num_classes

    names = list(generator.class_indices.keys())
    num_classes = len(names)
    if class_names == None:
        class_names = names
        return names

    elif len(class_names) != len(names):
        log('MISSMATCH BETWEEN THE NUMBER OF CLASSES IN A DATASET', 'error')
        log(f'Previously loaded classes: {class_names}', 'error')
        log(f'Currently loaded classes: {names}', 'error')
        # TODO Halt execution

    log(f'The following classes were loaded: {names}')
    return names


def init_generators(preprocessing_func, directory, rotation=0, width_shift=0, 
                    height_shift=0, shear=0, zoom=0, h_flip=False,
                    v_flip=False, suffle=False):
    # Setting up the data generator for all three phases
    # Training data gnerator
    datagen_train = ImageDataGenerator(preprocessing_function=PreProcess,
                                    rotation_range=rotation,
                                    width_shift_range=width_shift,
                                    height_shift_range=height_shift,
                                    shear_range=shear,
                                    zoom_range=zoom,
                                    horizontal_flip=h_flip,
                                    vertical_flip=v_flip,
                                    fill_mode='nearest')

    generator = datagen_train.flow_from_directory(directory=directory,
                                                        target_size=input_shape,
                                                        batch_size=batch_size,
                                                        shuffle=suffle)

    steps = generator.n / batch_size
    
    cls_train = generator.classes
    class_weight_train = compute_class_weight(
        class_weight='balanced', classes=np.unique(cls_train), y=cls_train)

    load_class_names(generator)

    return generator, steps, class_weight_train


'''  BUILDING THE MODEL  '''
def build_model():
    # Putting the model together
    if is_base:

        # Freezing the model
        loaded_model.trainable = False
        for layers in loaded_model.layers:
            layers.trainable = False

        # Adding other layers
        '''
        new_model = Sequential()
        new_model.add(loaded_model)
        new_model.add(tf.keras.layers.Flatten())
        new_model.add(Input(shape=(dense_amount)))
        '''

        # inp = Input(shape=(dense_amount))
        x = loaded_model.output
        x = tf.keras.layers.Flatten()(x)
        for i in range(dense_count):
            '''
            new_model.add(Dropout(0.5))
            new_model.add(Dense(dense_amount, activation='relu'))
            '''
            x = Dropout(0.5)(x)
            x = Dense(dense_amount, activation='relu')(x)

        # Output layer
        '''
        new_model.add(Dense(num_classes, activation='softmax'))
        '''
        out = Dense(num_classes, activation='softmax')(x)

        new_model = Model(inputs=loaded_model.input, outputs=out)

    else:
        new_model = loaded_model
    '''
    # Loading saved weights if any were specified
    if checkpoint_name != None:
        new_model.load_weights(str(full_path.joinpath(checkpoint_name)))
    '''
    if checkpoint_name != None:
        new_model.load_weights(pathlib.Path(checkpoint_path).joinpath(model_class_name).joinpath(checkpoint_name))
    new_model.summary()


    return new_model


def train_with_tf():
    # TODO develop
    pass


def train_with_keras(model, generator, generator_steps, generator_validation, generator_validation_steps, class_weight, epochs):
    accuracy = '0'
    log('Started training...')

    model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
    history = model.fit(x=generator, epochs=epochs, steps_per_epoch=generator_steps,
                callbacks=[
                    tensorboard_callback, checkpoint_callback], validation_data=generator_validation, validation_steps=generator_validation_steps,
                class_weight=(dict(enumerate(class_weight))))
    accuracy = f'{model.evaluate(generator_validation, steps=generator_validation_steps)[1]}'

    log(f'Test-set classification accuracy: {str(float(accuracy[:6]) * 100)}%')
    return history, accuracy


def train(model, generator, generator_steps, generator_validation, generator_validation_steps, class_weight, epochs, type_of_training='keras'):
    switcher = {
        'keras': train_with_keras,
        'tf': train_with_tf
    }

    return switcher[type_of_training](model, generator, generator_steps, generator_validation, generator_validation_steps, class_weight, epochs)
    

def export_model(model, accuracy):
    global timestamp_path

    # Exporting the trained model
    model.compile(optimizer=optimizer_finetuning,
                        loss=loss, metrics=metrics)
    
    timestamp_path = pathlib.Path(
            f'{saved_model_path}/{model_class_name}/{name}_a={float("%.2f" % float(accuracy)) * 100}%_e={epochs}_{timestamp}')  # Generate a unique file name
    log(f'Exporting trained model at {timestamp_path}')
    os.mkdir(f'{timestamp_path}')
    # new_model.save(f'{timestamp_path}')  # Save the model
    model.save(timestamp_path)  # Save the model

    # Exporting the class names into the model's directory
    with open(timestamp_path.joinpath('classes'), 'a') as file:
        for i in range(len(class_names)):
            file.write(f'{class_names[i].split("_")[0]}\n')


def export_as_tflite():
    # Saving converting the model to TFLite (optional)
    if save_as_lite:
        log('Saving model as TensorFlow Lite')
        converter = lite.TFLiteConverter.from_saved_model(
            str(pathlib.Path(timestamp_path)))
        tflite_model = converter.convert()
        tflite_file_name = f'{timestamp_path}/{name}_{timestamp}.tflite'
        with tf.io.gfile.GFile(tflite_file_name, 'wb') as f:
            f.write(tflite_model)
        log(f'Converted model was saved to {tflite_file_name}')

'''
def build_generators():
    global generator_train, steps_train, cls_weight_train, \
        generator_validation, steps_validation, cls_weight_validation, \
        generator_finetuning, steps_finetuning, cls_weight_finetuning
    
'''


def main():
    process_commands()
    if run:
        init_training_variables()
            
        generator_train, steps_train, cls_weight_train = init_generators(PreProcess, train_dir, 20, 0.1, 0.1, 0.1, [0.9, 1.1], False, True)
        generator_validation, steps_validation, cls_weight_validation = init_generators(PreProcess, validation_dir)
        if not skip_finetuning:
            generator_finetuning, steps_finetuning, cls_weight_finetuning = init_generators(PreProcess, finetuning_dir, 20, 0.1, 0.1, 0.1, [0.9, 1.1], False, True) 

        model = build_model()

        training_history, accuracy = train(model, generator_train, steps_train, generator_validation, steps_validation, cls_weight_train, epochs)
        training_history, accuracy = train(model, generator_finetuning, steps_finetuning, generator_validation, steps_validation, cls_weight_finetuning, epochs)

        # Printing the summary
        log(f'\n\tTRAINING FINISHED FOR {name.upper()}\nclasses({num_classes}): {class_names}\nepochs:\t{epochs} \
        | pictures:\t{steps_train * batch_size} | accuracy:\t{str(float(accuracy) * 100)[:6]}%')


if __name__ == '__main__':
    main()
