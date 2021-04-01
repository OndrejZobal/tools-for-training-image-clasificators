#!/bin/python

import os
import sys
import pathlib
import time
import threading
from shutil import copyfile
from shutil import copy
import code

training_name = 'training'
validation_name = 'validation'
finetuning_name = 'finetuning'

CALCULATE_THRESHOLD_AUTOMATICALLY = False
class_overfit_amount = 300
class_exclude_amount = 200
do_class_overfit = True
do_class_exclude = True

ratio = None
source = None
destination = None
do_symlink = True
do_print_dirs = True
thread_stop = False

source_dirs = []
source_files = []

sub_avg = []
sub_acc = []
amounts = []

phase = 0
progress_class = 0
progress_file = 0

DEBUG = False


def banner():
    print('''\
  ____                            
 (|   \                           
  |    | __, _|_  __,   ,   _ _|_ 
 _|    |/  |  |  /  |  / \_|/  |  
(/\___/ \_/|_/|_/\_/|_/ \/ |__/|_/
 ______        by Ondřej Zobal                  
(_) |                                            
   _|_  __   ,_    _  _  _    __, _|_  __   ,_   
  / | |/  \_/  |  / |/ |/ |  /  |  |  /  \_/  |  
 (_/   \__/    |_/  |  |  |_/\_/|_/|_/\__/    |_/
''')


# The map translating the single char arguments into full string arguments.
def log(message, level = 'info', start=' ', end='\n', hide_box=False):
    symbol = {
        'info': '*',
        'varning': '@',
        'error': '!',
    }

    box = f'[{symbol.get(level)}] ' if not hide_box else ''
    nl = '\n'
    print(f'{nl if level == "error" else ""}{start}{box}{message}',
        end=end)


# Exits the program prematurely and prints an error message
def stop(msg='AN ISSUE'):
    global thread_stop
    log(f'THE PROGRAM IS EXITTING DUE TO {msg.upper()}.')
    thread_stop = True
    exit()


# This function promts the user for setting up individual values.
def prompt(ratio=None, source=None, destination=None):
    # Getting the ratio
    if ratio == None:
        ratio = input('Input ratio [TRAINING, VALIDATION]: ')\
            .replace(' ', '').split(',')
        # Checking if there are only 1 or 2 elements in the array
        if 0 < len(ratio) < 3:
            # Converting every element to float
            for i in range(len(ratio)):
                try:
                    ratio[i] = float(ratio[i])
                # If the input cannot be converted to floats.
                except Exception as e:
                    log('Not a FLOAT!', 'error')
                    log(e, 'error')
                    ratio = None
                    # Recursively call this function to get a new input.
                    return prompt(ratio, source, destination)
            # If the sum of all the numbers is grater than one
            if sum(ratio) > 1:
                log('Sum of values is greater than 1!', 'error')
                ratio = None
                # Recursively call this function to get a new input.
                return prompt(ratio, source, destination)
        # If the input has a wrong format.
        else:
            log('Too few arguments or wrong formating!', 'error')
            ratio = None
            # Recursively call this function to get a new input.
            return prompt(ratio, source, destination)

    # Getting the soruce path.
    if source == None:
        source = pathlib.Path(input('Path to the source dir: '))
        if not os.path.isdir(source):
            source = None
            return prompt(ratio, source, destination)

    # Getting the destination path.
    if destination == None:
        destination = pathlib.Path(input('Path to the destination dir: '))
        if not os.path.isdir(source):
            source = None
            return prompt(ratio, source, destination)

    return ratio, source, destination


# Explores the direcotries and maps the file tree.
def map_dir():
    global source_dirs, source_files, source, sub_acc, sub_avg, \
        CALCULATE_THRESHOLD_AUTOMATICALLY, class_overfit_amount, \
        class_exclude_amount

    # TODO use generators.
    # Obtain directory listing.
    source_dirs = os.listdir(source)
    new_source_dirs = []

    # Puts all files into a table 'source_files'.
    dir_pointer = -1
    for i in range(len(source_dirs)):
        # Making sure all the 'file objects' lead to a dir and not some random
        # file in the 'source' dir.
        if os.path.isdir(source.joinpath(source_dirs[i])):
            # Adding a new list for each subdir that will contain it's files.
            source_files.append([])
            dir_pointer += 1
            # At the same time I am building a new source dir list, that
            # only contains actual directories. It's indexes will match the
            # the 'source_files' table.
            new_source_dirs.append(source_dirs[i])
            for j in os.listdir(source.joinpath(source_dirs[i])):
                # Making sure only files get added this time.
                if os.path.isfile(source.joinpath(source_dirs[i]).joinpath(j)):
                    source_files[dir_pointer].append(j)

    source_dirs = new_source_dirs

    # Calculeting the total amount of files
    total_sum = 0
    for i, dir in enumerate(source_dirs):
        total_sum += len(source_files[i])
    average = total_sum / len(source_files)

    # Automatic caluculation of the thresholds for overfitting and excluding
    # dataset classes based on the amount of samples provided.
    if CALCULATE_THRESHOLD_AUTOMATICALLY:
        if do_class_overfit and class_overfit_amount == 0:
            class_overfit_amount = total_sum / len(source_files)
        if do_class_exclude and class_exclude_amount == 0:
            class_exclude_amount = class_overfit_amount/3*2
    
    # Comunicating the information about the thresholds.
    log(f'The thresholds have been '+
        f'{"calculated" if CALCULATE_THRESHOLD_AUTOMATICALLY else "set"}'+
        ' to the following:')
    log(f'Threshold for overfitting dataset is {class_overfit_amount}.')
    log(f'Threshold for excluding dataset is {class_exclude_amount}.')

    # Computing the amounts of files needed to transfer for every class
    # according to the thresholds.
    for i, dir in enumerate(source_dirs):
        if len(source_files[i]) < class_exclude_amount:
            sub_acc.append(i)
            amounts.append(0)
        elif len(source_files[i]) < class_overfit_amount:
            sub_avg.append(i)
            amounts.append(int(average / len(source_files[i])))
        else:
            amounts.append(1)


# Creates direcory structure at the new location.
def make_dirs():
    global destination, source_dirs, training_name, validation_name, \
        finetuning_name, ratio

    log('Creating directory structure.')

    # Creating the root directory of the dataset.
    try:
        os.mkdir(destination)
        log('Destination directory created.')
    except FileExistsError:
        pass

    # Populating it with the 3 base dirs:
    # the training dir,
    try:
        os.mkdir(destination.joinpath(training_name))
    except FileExistsError:
        pass

    # the validation dir
    try:
        os.mkdir(destination.joinpath(validation_name))
    except FileExistsError:
        pass

    # and the finetuning dir.
    if len(ratio) == 2:
        try:
            os.mkdir(destination.joinpath(finetuning_name))
        except FileExistsError:
            pass

    # Making dirs for the individual classes.
    for i, directory in enumerate(source_dirs):
        if i in sub_acc:
            continue

        # Making subdirectories for each class.
        # in training.
        try:
            os.mkdir(destination.joinpath(training_name).joinpath(directory))
        except FileExistsError:
            pass

        # in validation.
        try:
            os.mkdir(destination.joinpath(validation_name).joinpath(directory))
        except FileExistsError:
            pass

        # in finetuning.
        if len(ratio) == 2:
            try:
                os.mkdir(destination.joinpath(
                    finetuning_name).joinpath(directory))
            except FileExistsError:
                pass


# Forwards creating a single file to the appropriate function.
def create(src, dest):
    global do_symlink
    if do_symlink:
        # Create a symbolic link
        try:
            os.symlink(src, dest)
        except FileExistsError as e:
            pass
        except OSError as e:
            stop(f'An OS error has occured: "{e}." !')

    else:
        # Make a copy of the file
        try:
            copyfile(src, dest)
        except FileExistsError as e:
            log(f'Cannot copy file {src}.', 'error')


# Populates a single dir with samples.
def make_dataset_dirs(path, destination_name, _range, index, amount):
    global progress_file
    progress = 0
    while True:
        for i in range(_range[0], _range[1]):
            if progress >= amount:
                return
            create(path.joinpath(source_files[index][i]), 
                destination.joinpath(destination_name, source_dirs[index], 
                f'{progress}-{source_files[index][i]}'))
            progress_file += 1
            progress += 1


# Initiates the population of all dirs of a one class with samples.
def make_dataset(index):
    global destination, source, source_dirs, source_files, ratio, progress_file

    dataset_path_src = source.joinpath(source_dirs[index])

    # Computes the real ranges based on the amount of samples and ratio.
    range_1 = int(len(source_files[index]) * ratio[0])
    range_2 = int(len(source_files[index]))
    ratio2_val = 1 - ratio[0]

    # TODO What the fuck was this thing supposed to do!
    do_finetuning = False

    # Check if finetuning category is enabled.
    if len(ratio) == 2:
        range_2 = range_1 + int(len(source_files[index]) * ratio[1])
        ratio2_val = ratio[1]
        do_finetuning = True

        # Finetuning
        make_dataset_dirs(dataset_path_src, finetuning_name,
            [range_2, len(source_files[index])], index, len(
            source_files[index]) * (1 - (ratio[0] + ratio[1])) \
            * amounts[index])

    # Training
    make_dataset_dirs(dataset_path_src, training_name, [0, range_1], index, 
        len(source_files[index]) * ratio[0] * amounts[index])

    # Validation
    make_dataset_dirs(dataset_path_src, validation_name, [range_1, range_2], 
        index, len(source_files[index]) * ratio2_val * amounts[index])


# Prints a list of the classes with the amounts of samples and their standing 
# according to the threshold.
def print_dirs():
    total_sum = 0
    for i, dir in enumerate(source_dirs):
        total_sum += len(source_files[i])
    average = total_sum / len(source_files)
    log(f'The average amount of files is {int(average)}.\n')

    log(f'List of directories:\n\t', end='')
    for i, dir in enumerate(source_dirs):
        log(f'{dir}\t\t\thas {len(source_files[i])} samples', end='')
        if len(source_files[i]) < average/3*2:
            log(' (BELLOW ACCEPTABLE.)', start='', hide_box=True)
        elif len(source_files[i]) < average:
            log(' (Bellow average.)', start='', hide_box=True)
        else:
            log('.', start='', hide_box=True)
        log('', hide_box=True, end='\t')

    log(f'There are {len(source_dirs)-len(sub_acc)-len(sub_avg)}'+
        ' heathy datasets.')
    log(f'There are {len(sub_avg)} files that are bellow average.')
    log(f'There are {len(sub_acc)} files that have issufitient amount'+
        ' of sampless.\n')


# A function that displays the proggress bar and also prints some other things.
# It is ment to run in a separate thread.
def progress_bar():
    dirs = len(source_dirs)
    while phase == 0:
        time.sleep(0.01)

    # Printing in Python is inconsistent sometimes.
    # Therefore I have to print the directory listing in a separate thread.

    while phase == 1 and not thread_stop:
        time.sleep(1.33)
        # Dumps the list of dirs.
        if DEBUG:
            string = 'Mapped directory structure: '
            if dirs < len(source_dirs):
                dirs = len(source_dirs)
                # Delete previous line
                for i in source_dirs:
                    string += f'{i}, '
            log(string)

    # TODO remove from this function.
    files_total = 0
    for i, obj in enumerate(source_files):
        files_total += len(obj) * amounts[i]
    log(f'Total files found: {files_total}.')

    log('\n' * 3, start='', hide_box=True)
    # Showing and updating the progress bar.
    while phase == 2 and not thread_stop:
        # Putting the cursor three lines up.
        sys.stdout.write("\033[F" * 3)
        # Giving the CUP a break before redrawing.
        time.sleep(0.05)
        # Computing the current percentage of progress.
        percentage_class = float(progress_class) / len(source_files) * 100
        percentage_file = float(progress_file) / files_total * 100
        log(f'{"Linking" if do_symlink else "Copying"} new dataset'+\
            f' structure at {destination}')
        
        # Preparing the string that will be printed.
        string = f'\t[~] Class progress\t[{"█" * (int(percentage_class))}'+\
            f'{" " * (100 - int(percentage_class))}] {int(percentage_class)}%'\
            + f' - {int(progress_class)}/{len(source_files)}'+\
            f' ({source_dirs[progress_class-1]}){" "*10}\n'

        string += f'\t[~] File progress\t[{"█" * (int(percentage_file))}'+\
        f'{" " * (100 - int(percentage_file))}] {int(percentage_file)}% -'+\
        f' {int(progress_file)}/{files_total}{" "*10}\n'
        log(string, end='', start='', hide_box=True)
    
    log('Done formating dataset.')
    return


def main():
    global ratio, source, destination, source_dirs, training_name, \
        validation_name, finetuning_name, phase, progress_class

    # Greeter banner
    banner()

    # Getting the basic parameters
    ratio, source, destination = prompt(
        [0.8], pathlib.Path('/winD/Houby/'), pathlib.Path('./dataset'))

    # Display the progress bar
    prog_bar = threading.Thread(target=progress_bar)
    prog_bar.start()

    phase = 1
    log('Mapping direcotries.')
    map_dir()

    # Prints all the directories with the amount of pictures in them
    if do_print_dirs:
        print_dirs()

    # Create the directory structure
    phase = 2
    make_dirs()

    # Copy files into the new dirs
    for i in range(len(source_dirs)):
        make_dataset(i)
        progress_class += 1

    phase = 3
    prog_bar.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop('A KEYBOARD INTERRUPTION')
