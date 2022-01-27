import pandas as pd
import numpy as np


def get_dataset_df(args):
        file_path = f'Data/{args.dataset}/{args.dataset}/ratings.dat'
        df = pd.read_csv(file_path, sep='::', header=None, engine='python')
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

def save_dataset_df(args, df, tail):
    file_path = f'Data/{args.dataset}/{args.dataset}-{tail}/ratings.dat'
    np.savetxt(file_path, df, fmt='%s', delimiter='::')