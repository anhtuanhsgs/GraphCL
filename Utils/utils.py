import time
import os

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
