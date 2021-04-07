#!/bin/python


# Begin license text.
# 
# Copyright 2020 Ondřej Zobal
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


import os
import sys
import pathlib
import time
import threading
from shutil import copyfile

# ADJUSTABLE VARIABLES.
# Change the following group of variables to alter the behavior of the script.

# The categories into wich the dataset will be divided.
categories = [
    'training',
    'validation',
    'finetuning',
    ]

# Ratio to split values between categories.
ratio = [
    0.6,
    0.2,
    0.2,
]

# When true 'class_overfit_amount' or 'class_exclude_amount' will be replaced 
# with automatically calculated value if set to 0.
calculate_threshold_automatically = False 
# Zero means don't overfit, if 'calculate_threshold_automatically' is False.
class_overfit_amount = 300
# Zero means don"t exclude, if 'calculate_threshold_automatically' is False.
class_exclude_amount = 200

# Path to the source directory.
source = None
# Path to the directory where the formated dataset will be created.
destination = None
# If true will use symlink, otherwise copying will be employed.
do_symlink = True

# NONADJUSTABLE VARIABLES.
# These variables will be set by the program.

# Will be filled with the class names.
source_dirs = []
# A table, will contain lists with files.
source_files = []

# List of indexes of classes that have to be overfitted.
sub_avg = []
# List of indexes of classes that have to be excluded.
sub_acc = []
# List of amounts of will that will need to be copied
amounts = []

# Progress of creating classes.
progress_class = 0
# Progress of creating files.
progress_file = 0
# Bollean indicating the thread should exit.
thread_stop = False

# May activate some debug features.
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
        'warning': 'W',
        'error': '!',
    }

    box = f'[{symbol.get(level)}] ' if not hide_box else ''
    nl = '\n'
    print(
        f'{nl if level == "error" else ""}{start}{box}{message}',
        end=end)


# Exits the program prematurely and prints an error message.
def stop(msg='AN ISSUE'):
    global thread_stop
    log(f'THE PROGRAM IS EXITTING DUE TO: {msg.upper()}', 'error')
    thread_stop = True
    exit()


# Print a help text.
def arg_help(next_arg):
    log('''You can use these flags:
    -h\tDisplays this message.
    -e\tMinimum amount of samples, before the dataset will be excluded.
    -o\tMinimum amount of samples, before the dataset will be overfitted.
    -S\tFlips calculate_thresholds_automatically. If true exclusion and 
        overfitting amounts will be calculated if they have been set to 0.
    -r\tThe ratios between the categories separated by commas. (Ex. 0.6,0.4)
    -c\tThe names of categories separated by commas. (Ex. train,validation)
    -s\tThe directory of the source.
    -d\tThe directory of the destination.
    -S\tFlips do_symlink variable. If true links will be used instead of copying.
    ''')
    exit()
    return False


# Sets the ratio
def arg_ratio(next_arg):
    global ratio
    string_list = next_arg.split(',')
    ratio = []
    for x in string_list:
        ratio.append(float(x))
        
    # If the sum of all the numbers is grater than one
    if sum(ratio) > 1:
        stop('Sum of ratio is greater than 1!')

    return True


# Setts the source directory
def arg_source(next_arg):
    global source
    source = next_arg
    return True


# Setts the destination directory
def arg_destination(next_arg):
    global destination
    destination = pathlib.Path(next_arg)
    return True


# Flips value in 'do_symlink'.
def arg_symlink(next_arg):
    global do_symlink
    do_symlink = not do_symlink
    return False


# Sets the overfitting threshold.
def arg_overfit_amount(next_arg):
    global class_overfit_amount
    try:
        class_overfit_amount = int(next_arg)
    except:
        stop('Invalid command line argument passed for overfitting amount.')
    return True


# Sets the overfitting threshold.
def arg_exclude_amount(next_arg):
    global class_exclude_amount
    try:
        class_exclude_amount = int(next_arg)
    except:
        stop('Invalid command line argument passed for exclusion amount.')
    return True

# Flips value in 'calculate_threshold_automatically'.
def arg_calculate_threshold(next_arg):
    global calculate_threshold_automatically
    calculate_threshold_automatically = not calculate_threshold_automatically
    return False

# Sets a name for the output categories
def arg_category(next_arg):
    global categories
    categories = next_arg.split(',')
    return True


# Dict mapping the short forms of flags to the long ones.
char_arg_map = {
    # Short form | Long form
    'C': 'calculate-thresholds',
    'e': 'exclude',
    'o': 'overfit',
    'r': 'ratio',
    's': 'source',
    'd': 'destination',
    'S': 'symlink',
    'c': 'categories',
    'h': 'help'}

# Maps the long name of a flag to a argument function
arg_dict = {
    # Key | Function
    'calculate-thresholds': arg_calculate_threshold,
    'exclude': arg_exclude_amount,
    'overfit': arg_overfit_amount,
    'ratio': arg_ratio,
    'source': arg_source,
    'destination': arg_destination,
    'symlink': arg_symlink,
    'categories': arg_category,
    'help': arg_help, }


# Converts single char arguments into a full argument and calls
# the processing function.
def process_1char_arg(char, next_arg):
    try:
        # return process_arg(char_arg_map.get(char), next_arg)
        return arg_dict[char_arg_map[char]](next_arg)
    except Exception as e:
        log(f'\nInvalid single dash argument was given:\n\t{e}', 'error')


# Process command line arguments args
def process_commands():
    if len(sys.argv) <= 0:
        return
    # Set to True when flag that requires aditional argument after
    skip = False
    for arg in range(len(sys.argv)):
        skip = False
        if skip:
            continue
        if (sys.argv[arg][0] == '-'):
            next_arg = ''
            if len(sys.argv) + 1 >= arg:
                try:
                    next_arg = sys.argv[arg + 1]
                except:
                    pass
            # Handeling 'one dash per onle letter' syntax.
            # This will permit passing one aditional parameter
            if len(sys.argv[arg]) == 2:
                skip = process_1char_arg(sys.argv[arg][1], next_arg)
            # Long arguments
            elif len(sys.argv[arg]) > 3:
                # Handeling 'double dash, whole word! syntax.
                # This will permit passing aditional parameters
                if sys.argv[arg][1] == '-':
                    skip = arg_dict[sys.argv[arg][2:]](next_arg)
        else:
            # Consider the possibility of a default argument
            pass


# This function prompts the user for setting up individual values.
def prompt(ratio=None, source=None, destination=None):
    # Getting the ratio
    if ratio == None:
        temp = input('Input ratio [TRAINING, VALIDATION]: ')\
            .replace(' ', '').split(',')
        ratio = []
        for i in temp:
            try:
                ratio.append(float(i))
            except:
                log('Not a FLOAT!', 'error')
                ratio = None

        if len(ratio) != len(categories) or len(ratio) != len(categories)-1:
            ratio = None
        else:
            log('all fine', 'warning')

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

    # Getting the source path.
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


# Checks if the values in the config file make sense.
def check_validity_of_configuration():
    global ratio
    # Calculating the last number of the ratio, in case the user was lazy.
    if len(ratio) == len(categories) - 1:
        ratio.append(1-sum(ratio))
    
    elif len(ratio) != len(categories):
        stop('Mismatch between the amount of categories and ratio')


def log_info():
    log(f'Sourcing dataset from {os.path.abspath(source)}.')
    log(f'Makeing a new {"linked" if do_symlink else "copying"} dataset at '
        + f'{os.path.abspath(destination)}.')
    log('', start='', hide_box=True)
    log(f'The dataset will be split according to the following:')
    for i, category in enumerate(categories):
        log(f'{i}. {category}:\t{ratio[i]}')
    log('', start='', hide_box=True)

# Explores the direcotries and maps the file tree.
def map_dir():
    global source_dirs, source_files, source, sub_acc, sub_avg, \
        calculate_threshold_automatically, class_overfit_amount, \
        class_exclude_amount

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
    for i, directories in enumerate(source_dirs):
        total_sum += len(source_files[i])
    average = total_sum / len(source_files)

    # Automatic calculation of the thresholds for overfitting and excluding
    # dataset classes based on the amount of samples provided.
    if calculate_threshold_automatically:
        if class_overfit_amount == 0:
            class_overfit_amount = total_sum / len(source_files)
        if class_exclude_amount == 0:
            class_exclude_amount = class_overfit_amount/3*2
    
    # Comunicating the information about the thresholds.
    log(
        f'The thresholds have been '
        + f'{"calculated" if calculate_threshold_automatically else "set"}'
        + ' to the following:')
    log(f'Threshold for overfitting dataset is {class_overfit_amount}.')
    log(f'Threshold for excluding dataset is {class_exclude_amount}.')

    # Computing the multiplier of amounts of files needed to transfer for 
    # every class # according to the thresholds.
    for i, direcotries in enumerate(source_dirs):
        if class_exclude_amount != 0 \
                and len(source_files[i]) < class_exclude_amount:
            sub_acc.append(i)
            amounts.append(0)
        elif class_exclude_amount != 0 \
                and len(source_files[i]) < class_overfit_amount:
            sub_avg.append(i)
            amounts.append(int(average / len(source_files[i])))
        else:
            amounts.append(1)


# Creates direcory structure at the new location.
def make_dirs(destination, dir_list):

    log('Creating directory structure.')

    # Creating the root directory of the dataset.
    try:
        os.mkdir(destination)
        log('Destination directory created.')
    except FileExistsError:
        pass

    for i in dir_list:
        try:
            os.mkdir(destination.joinpath(i))
        except FileExistsError:
            pass
        except Exception:
            stop(f'Couldn\'t create directory for {i}!')

        # Making dirs for the individual classes.
        for index, directory in enumerate(source_dirs):
            # Skip excluded datasets.
            if index in sub_acc:
                continue

            # Creating the class direcotories.
            try:
                os.mkdir(destination.joinpath(i).joinpath(directory))
            except FileExistsError:
                pass
            except Exception:
                stop(f'Couldn\'t create directory for {directory} in {i}!')


# Forwards creating a single file to the appropriate function.
def create(src, dest):
    if do_symlink:
        # Create a symbolic link
        try:
            os.symlink(src, dest)
        except FileExistsError as e:
            pass
        except OSError as e:
            stop(f'An OS error has ocurred: "{e}." !')

    else:
        # Make a copy of the file
        try:
            copyfile(src, dest)
        except FileExistsError as e:
            log(f'Cannot copy file {src}.', 'error')


# Populates a single dir with samples.
def make_dataset_dirs(path, destination_name, occurred, index, amount):
    global progress_file
    progress = 0
    while True:
        for i in range(occurred[0], occurred[1]):
            if progress >= amount:
                return
            create(
                path.joinpath(source_files[index][i]), 
                destination.joinpath(destination_name, source_dirs[index], 
                f'{progress}-{source_files[index][i]}'))
            progress_file += 1
            progress += 1


# Initiates the population of all dirs of a one class with samples.
def make_dataset(index):
    global ratio

    dataset_path_src = source.joinpath(source_dirs[index])

    ranges = [0]

    # Computes the real ranges based on the amount of samples and ratio.
    for i, value in enumerate(ratio):
        # log(f'randomass var: {value}', 'error')
        ranges.append(
            ranges[len(ranges)-1] + int(len(source_files[index]) * value))

        make_dataset_dirs(
            dataset_path_src, categories[i], [ranges[i], ranges[i+1]],
            index, (ranges[i+1] - ranges[i]) * amounts[index])


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
            log(' (excluding)', start='', hide_box=True)
        elif len(source_files[i]) < average:
            log(' (overfitting)', start='', hide_box=True)
        else:
            log('', start='', hide_box=True)
        log('', hide_box=True, end='\t')

    log('\n' * 2, start='', end='', hide_box=True)

    log(
        f'There are {len(source_dirs)-len(sub_acc)-len(sub_avg)}'
        + ' heathy datasets.')
    log(f'There are {len(sub_avg)} files that are bellow average.')
    log(
        f'There are {len(sub_acc)} files that have insufficient amount'
        + ' of samples.\n')


# A function that displays the progress bar and also prints some other things.
# It is meant to run in a separate thread.
def progress_bar():
    dirs = len(source_dirs)

    # Printing in Python is inconsistent when doing a lot of processing
    # Therefore I have to print the directory listing in a separate thread.
    files_total = 0
    for i, obj in enumerate(source_files):
        files_total += len(obj) * amounts[i]
    log(f'Total files found: {files_total}.')

    log('\n', start='', hide_box=True)

    # Showing and updating the progress bar.
    while not thread_stop:
        # Computing the current percentage of progress.
        percentage_class = float(progress_class) / len(source_files) * 100
        percentage_file = float(progress_file) / files_total * 100
        log(
            f'{"Linking" if do_symlink else "Copying"} new dataset'
            + f' structure at {destination}')
        
        # Preparing the string that will be printed.
        string = f'\t[~] Class progress\t[{"█" * (int(percentage_class))}'\
            + f'{" " * (100 - int(percentage_class))}]'\
            + f'{int(percentage_class)}%'\
            + f' - {int(progress_class)}/{len(source_files)}'\
            + f' ({source_dirs[progress_class-1]}){" "*10}\n'

        string += f'\t[~] File progress\t[{"█" * (int(percentage_file))}'\
            + f'{" " * (100 - int(percentage_file))}]'\
            +  f'{int(percentage_file)}% -'\
            + f' {int(progress_file)}/{files_total}{" "*10}\n'
        log(string, end='', start='', hide_box=True)
    
        # Giving the CPU a break before redrawing.
        time.sleep(0.01)
        # Putting the cursor three lines up.
        sys.stdout.write("\033[F" * 3)

    log(f'Finished formating dataset at \'{os.path.abspath(destination)}\'.' 
        + ' '*20)


def main():
    global ratio, source, destination, source_dirs, progress_class, thread_stop

    # Processes command line arguments.
    process_commands()

    # Greeter banner.
    banner()

    # Getting the basic parameters.
    ratio, source, destination = prompt(
        ratio, pathlib.Path('/winD/Houby/'), pathlib.Path('./dataset'))

    # Making sure all the config makes sense.
    check_validity_of_configuration()

    # Printing a summary of configuration.
    log_info()

    # Mapping files structure of the source dir.
    log('Mapping direcotries...\n')
    map_dir()

    # Prints all the directories with the amount of pictures in them.
    print_dirs()

    # Display the progress bar.
    prog_bar = threading.Thread(target=progress_bar)
    prog_bar.start()
    
    # Create the directory structure.
    make_dirs(destination, categories)

    # Copy files into the new dirs.
    for i in range(len(source_dirs)):
        make_dataset(i)
        progress_class += 1

    # Stopping the progress bar.
    thread_stop = True
    prog_bar.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop('A KEYBOARD INTERRUPTION')
