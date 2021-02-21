#!/bin/python

import os
import sys
import pathlib
import time
import threading
from shutil import copyfile
from shutil import copy
import code

training_name = 'Training'
validation_name = 'Validation'
finetuning_name = 'Finetuning'

CALCULATE_THRESHOLD_AUTOMATICALLY = False
class_overfit_amount = 300
class_exclude_amount = 200

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


def banner():
    print('\
  ____                            \n\
 (|   \                           \n\
  |    | __, _|_  __,   ,   _ _|_ \n\
 _|    |/  |  |  /  |  / \_|/  |  \n\
(/\___/ \_/|_/|_/\_/|_/ \/ |__/|_/\n\
 ______                                         \n\
(_) |                                            \n\
   _|_  __   ,_    _  _  _    __, _|_  __   ,_   \n\
  / | |/  \_/  |  / |/ |/ |  /  |  |  /  \_/  |  \n\
 (_/   \__/    |_/  |  |  |_/\_/|_/|_/\__/    |_/\n\
    by Ondřej Zobal\n\
    ')


def stop(msg='AN ISSUE'):
    global thread_stop
    print(f'\nTHE PROGRAM IS EXITTING DUE TO {msg.upper()}.\n')
    thread_stop = True
    exit()


def prompt(ratio=None, source=None, destination=None):
    # Getting the ratio
    if ratio == None:
        ratio = input('Input ratio [TRAINING, VALIDATION]: ')\
            .replace(' ', '').split(',')
        if 0 < len(ratio) < 3:  # Checking if there are only 1 or 2 elements in the array
            for i in range(len(ratio)):     # Converting every element to float
                try:
                    ratio[i] = float(ratio[i])
                except Exception as e:
                    print('Not a FLOAT!')
                    print(e)
                    ratio = None
                    return prompt(ratio, source, destination)
            if sum(ratio) > 1:  # If the sum of all the numbers is grater than one
                print('Sum of values is greater than 1!')
                ratio = None
                return prompt(ratio, source, destination)
        else:
            print('Too few arguments or wrong formating!')
            ratio = None
            return prompt(ratio, source, destination)

    # Getting the soruce path
    if source == None:
        source = pathlib.Path(input('Path to the source dir: '))
        if not os.path.isdir(source):
            source = None
            return prompt(ratio, source, destination)

    # Getting the destination path
    if destination == None:
        destination = pathlib.Path(input('Path to the destination dir: '))
        if not os.path.isdir(source):
            source = None
            return prompt(ratio, source, destination)

    return ratio, source, destination


def map_dir():
    global source_dirs, source_files, source, sub_acc, sub_avg, CALCULATE_THRESHOLD_AUTOMATICALLY, class_overfit_amount, class_exclude_amount

    # Todo use generators
    # rel_path = pathlib.Path()
    source_dirs = os.listdir(source)
    new_source_dirs = []

    dir_pointer = -1
    for i in range(len(source_dirs)):
        if os.path.isdir(source.joinpath(source_dirs[i])):
            source_files.append([])
            dir_pointer += 1
            new_source_dirs.append(source_dirs[i])
            for j in os.listdir(source.joinpath(source_dirs[i])):
                if os.path.isfile(source.joinpath(source_dirs[i]).joinpath(j)):
                    source_files[dir_pointer].append(j)

    source_dirs = new_source_dirs

    total_sum = 0
    for i, dir in enumerate(source_dirs):
        total_sum += len(source_files[i])
    average = total_sum / len(source_files)

    if CALCULATE_THRESHOLD_AUTOMATICALLY and class_overfit_amount == 0 and class_exclude_amount == 0:
        class_overfit_amount = total_sum / len(source_files)
        class_exclude_amount = class_overfit_amount/3*2

    for i, dir in enumerate(source_dirs):
        if len(source_files[i]) < class_exclude_amount:
            sub_acc.append(i)
            amounts.append(0)
        elif len(source_files[i]) < class_overfit_amount:
            sub_avg.append(i)
            amounts.append(int(average / len(source_files[i])))
        else:
            amounts.append(1)


def make_dirs():
    global destination, source_dirs, training_name, validation_name, finetuning_name, ratio

    try:
        os.mkdir(destination)
        print('Destination directory created.')
    except FileExistsError as e:
        pass
    try:
        os.mkdir(destination.joinpath(training_name))
    except FileExistsError as e:
        pass
    try:
        os.mkdir(destination.joinpath(validation_name))
    except FileExistsError as e:
        pass

    if len(ratio) == 2:
        try:
            os.mkdir(destination.joinpath(finetuning_name))
        except FileExistsError as e:
            pass

    for i, directory in enumerate(source_dirs):
        if i in sub_acc:
            continue

        # Making the main directories
        # Training
        try:
            os.mkdir(destination.joinpath(training_name).joinpath(directory))
        except FileExistsError as e:
            pass

        # Validation
        try:
            os.mkdir(destination.joinpath(validation_name).joinpath(directory))
        except FileExistsError as e:
            pass

        # Finetuning
        if len(ratio) == 2:
            try:
                os.mkdir(destination.joinpath(
                    finetuning_name).joinpath(directory))
            except FileExistsError as e:
                pass


def create(src, dest):
    global do_symlink
    if do_symlink:
        try:
            os.symlink(src, dest)
        except FileExistsError as e:
            pass
        except OSError as e:
            print(f'! An OS error has occured: "{e}." !\n')
            stop()

    else:
        try:
            copyfile(src, dest)
        except FileExistsError as e:
            pass


def make_dataset_dirs(path, destination_name, rng, index, amount):
    global progress_file
    progress = 0
    while True:
        for i in range(rng[0], rng[1]):
            if progress >= amount:
                return
            create(path.joinpath(source_files[index][i]),
                   destination.joinpath(destination_name).joinpath(source_dirs[index]).joinpath(f'{progress}-{source_files[index][i]}'))
            progress_file += 1
            progress += 1


def make_dataset(index):
    global destination, source, source_dirs, source_files, ratio, progress_file

    # dataset_path_des = destination.joinpath(source_dirs[index])
    dataset_path_src = source.joinpath(source_dirs[index])

    range_1 = int(len(source_files[index]) * ratio[0])
    range_2 = int(len(source_files[index]))
    ratio2_val = 1 - ratio[0]
    do_finetuning = False

    if len(ratio) == 2:
        range_2 = range_1 + int(len(source_files[index]) * ratio[1])
        ratio2_val = ratio[1]
        do_finetuning = True

        # Finetuning
        '''
        if do_finetuning:
            for i in range(range_2, len(source_files[index])):
                create(dataset_path_src.joinpath(source_files[index][i]),
                    destination.joinpath(finetuning_name).joinpath(source_dirs[index]).joinpath(source_files[index][i]))
                progress_file += 1
        '''
        make_dataset_dirs(dataset_path_src, finetuning_name, [range_2, len(source_files[index])], index, len(
            source_files[index]) * (1 - (ratio[0] + ratio[1])) * amounts[index])

    # Training
    '''
    for i in range(range_1):
        create(dataset_path_src.joinpath(source_files[index][i]),
               destination.joinpath(training_name).joinpath(source_dirs[index]).joinpath(source_files[index][i]))
        progress_file += 1
    '''
    make_dataset_dirs(dataset_path_src, training_name, [0, range_1], index, len(
        source_files[index]) * ratio[0] * amounts[index])

    # Validation
    '''
    for i in range(range_1, range_2):
        create(dataset_path_src.joinpath(source_files[index][i]),
               destination.joinpath(validation_name).joinpath(source_dirs[index]).joinpath(source_files[index][i]))
        progress_file += 1
    '''
    make_dataset_dirs(dataset_path_src, validation_name, [range_1, range_2], index, len(
        source_files[index]) * ratio2_val * amounts[index])


def print_dirs():
    total_sum = 0
    for i, dir in enumerate(source_dirs):
        total_sum += len(source_files[i])
    average = total_sum / len(source_files)
    print(f'\nThe average amount of files is {int(average)}.\n')

    print(f'List of directories')
    for i, dir in enumerate(source_dirs):
        print(f'{dir}\t\t\t\t\thas {len(source_files[i])} samples', end='')
        if len(source_files[i]) < average/3*2:
            print(' (BELLOW ACCEPTABLE.)')
        elif len(source_files[i]) < average:
            print(' (Bellow average.)')
        else:
            print('.')

    print(
        f'\nThere are {len(source_dirs)-len(sub_acc)-len(sub_avg)} heathy datasets.')
    print(f'There are {len(sub_avg)} files that are bellow average.')
    print(
        f'There are {len(sub_acc)} files that have issufitient amount of sampless.\n')


def progress_bar():
    dirs = len(source_dirs)
    while phase == 0:
        time.sleep(0.01)

    print('\n')
    while phase == 1 and not thread_stop:
        sys.stdout.write("\033[F"*1)  # Cursor up one line
        time.sleep(1.33)
        string = 'Mapping the directory structure: '
        if dirs < len(source_dirs):
            dirs = len(source_dirs)
            # Delete previous line
            for x, i in enumerate(source_dirs):
                string += f'{i}, '
        print(string)

    files_total = 0
    for i, obj in enumerate(source_files):
        files_total += len(obj) * amounts[i]
    print(files_total)

    print('\n' * 3)
    while phase == 2 and not thread_stop:
        sys.stdout.write("\033[F" * 3)
        time.sleep(0.001)
        percentage_class = float(progress_class) / len(source_files) * 100
        percentage_file = float(progress_file) / files_total * 100
        string = f'{"Linking" if do_symlink else "Copying"} new dataset structure at {destination}\n'
        string += f'Class progress\t[{"█" * (int(percentage_class))}{" " * (100 - int(percentage_class))}] {int(percentage_class)}% - {int(progress_class)}/{len(source_files)} ({source_dirs[progress_class-1]})\n'
        string += f'File progress\t[{"█" * (int(percentage_file))}{" " * (100 - int(percentage_file))}] {int(percentage_file)}% - {int(progress_file)}/{files_total}\n'
        print(string, end='')

    print('Done.')
    return


def main():
    global ratio, source, destination, source_dirs, training_name, validation_name, finetuning_name, phase, progress_class

    # Greeter banner
    banner()

    # Getting the basic parameters
    ratio, source, destination = prompt(
        [0.8], pathlib.Path('/winD/Houby/'), pathlib.Path('./dataset'))

    print(ratio)

    # Display the progress bar
    prog_bar = threading.Thread(target=progress_bar)
    prog_bar.start()

    phase = 1
    print('mapping dirs')
    map_dir()
    print('done mapping dirs')

    # Prints all the directories with the amount of pictures in them
    if do_print_dirs:
        print_dirs()
    print('Done printing dirs')

    # Create the directory structure
    phase = 2
    make_dirs()
    print('making dirs')

    # Copy files into the new dirs
    for i in range(len(source_dirs)):
        make_dataset(i)
        progress_class += 1
    print('done making dataset')

    phase = 3
    prog_bar.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop('A KEYBOARD INTERRUPTION')
