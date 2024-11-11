import time
import torch
from sympy.codegen.ast import float32
from torch.utils.data import Dataset
import pandas as pd
from typing import Final
import numpy as np
from threading import Thread, Lock
import multiprocessing




class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        print("Loading data using single process...")
        start_time = time.time()

        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file):
        print("Loading data using multi process...")
        start_time = time.time()


        # Create empty dataframes arrays with the correct size
        FEATURE_COLUMNS: Final[list[str]] = "x1 x2 x3".split()
        headers = pd.read_csv(csv_file, nrows=0).columns.tolist()
        column_types = { i: float if c in FEATURE_COLUMNS else int for i, c in enumerate(headers)}
        with open(csv_file, 'r') as f:
            row_count = sum(1 for _ in f) - 1
        self.data = pd.DataFrame(index=range(row_count), columns=headers)
        for c in FEATURE_COLUMNS:
            self.data[c] = pd.Series(np.nan, index=self.data.index, dtype=float)
        self.data["label"] = pd.Series(0, index=self.data.index, dtype=int)
        self.features = torch.zeros((row_count, len(FEATURE_COLUMNS)))
        self.labels = torch.zeros((row_count,))

        def load_data_block(lock: Lock, start_row: int, end_row: int) -> None:
            n_rows = end_row - start_row
            block = pd.read_csv(csv_file, skiprows=range(start_row + 1), nrows=n_rows, header=None,
                                names=headers, dtype=column_types)

            features_block = torch.FloatTensor(block[FEATURE_COLUMNS].values)
            labels_block = torch.LongTensor(block["label"].values.flatten())

            with lock:
                # Copy to the merged global data
                self.features[start_row:end_row] = features_block
                self.labels[start_row:end_row] = labels_block
                self.data.iloc[start_row:end_row] = block.values

        # Calculate the row ranges using row_count and number of CPU processors
        def div_up(x: int, divisor: int) -> int:
            return (x + divisor - 1) // divisor

        num_threads = multiprocessing.cpu_count()
        block_row_count = div_up(row_count, num_threads - 1)
        rows = [min(block_row_count * i, row_count) for i in range(num_threads)]

        # Launch the thread workers with the lock
        lock = Lock()
        def start_thread(start_row: int, end_row: int) -> Thread:
            td = Thread(target=load_data_block, args=(lock, start_row, end_row))
            td.start()
            return td

        threads = [start_thread(s, e) for s,e in zip([0] + rows[:-1], rows)]
        for t in threads: t.join()

        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

def load_data_loaders():
    """ Used for unit testing of data loaders"""
    CSV_PATH: Final[str] = "provided/3d_data.csv"
    return SingleProcessDataset(CSV_PATH), MultiProcessDataset(CSV_PATH)
