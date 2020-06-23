import csv
import numpy as np
import argparse
from datetime import datetime, timedelta

def file_writer(results_file, meta_info, acc, loss_val):
    lr, epochs, seed = meta_info
    params = [" LR ", str(lr), " SEED ", str(seed), " EPOCHS ", str(epochs)]

    with open(results_file, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        writer.writerow(['loss_val: ', str(loss_val)])
        writer.writerow(['acc_val: ', str(acc)])
        writer.writerow("")

        csvFile.flush()

    csvFile.close()
    return

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-inp_dir", type=str, default='data/pkl_data/')
    ap.add_argument("-seed", type=int, default=0)
    args = ap.parse_args()
    return args.inp_dir, args.seed
