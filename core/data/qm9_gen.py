# from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate
import torch
from functools import partial
import numpy as np
import os
import pickle
import pandas as pd
import math

from core.data.qm9 import QM9
# from core.data.prefetch import PrefetchLoader
# import core.utils.ctxmgr as ctxmgr

# from qm9 import QM9
# from .prefetch import PrefetchLoader
# from ..utils.ctx`mgr import ctxmgr

from absl import logging
import time


def remove_mean(pos, dim=0):
    mean = torch.mean(pos, dim=dim, keepdim=True)
    pos = pos - mean
    return pos


def _make_global_adjacency_matrix(n_nodes, device="cpu"):
    # device = "cpu"
    row = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, -1, 1)
        .repeat(1, 1, n_nodes)
        .to(device=device)
    )
    col = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, 1, -1)
        .repeat(1, n_nodes, 1)
        .to(device=device)
    )
    full_adj = torch.concat([row, col], dim=0)
    diag_bool = torch.eye(n_nodes, dtype=torch.bool).to(device=device)
    return full_adj, diag_bool


class QM9Gen(DataLoader):
    num_atom_types = 5

    def __init__(
        self, datadir, batch_size, n_node_histogram, device="cpu", **kwargs
    ) -> None:
        print(f"datadir is: {datadir}")
        self.datadir = datadir
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.device = device
        self.max_n_nodes = len(n_node_histogram) + 10
        self.full_adj, self.diag_bool = _make_global_adjacency_matrix(self.max_n_nodes, device)
        ds = QM9(
            root=datadir,
            transform=self.transform,
            split=kwargs.get("split", "train"),
        )
               
        sample_order_file = f"{datadir}/batchsize{batch_size}.pkl"
        data_list_file = f"{datadir}/Databatchsize{batch_size}.pkl"
        ds_batch = self.build_ds_batch(ds, self.batch_size, sample_order_file, data_list_file)
        self.ds = ds_batch
        super().__init__(self.ds, batch_size=1, shuffle=kwargs.get("shuffle", True))
    
    def build_ds_batch(self, ds_ori, batch_size, sample_order_file=None, data_list_file=None):

        if sample_order_file is not None and os.path.exists(sample_order_file):
            with open(sample_order_file, 'rb') as file:
                sample_order = pickle.load(file)
        else:  
            len_ds = []
            for idx in range(len(ds_ori)):
                data_val = ds_ori[idx]
                len_ds.append([idx, data_val.x.shape[0]])
            len_ds_pd = pd.DataFrame(len_ds, columns=["index", "length"])
            count_val = len_ds_pd.groupby("length")
            sample_order = []
            for seq_len, len_df in count_val:
                num_batches = math.floor(len(len_df) / batch_size)
                for i in range(num_batches):
                    batch_df = len_df.iloc[i*batch_size:(i+1)*batch_size]
                    batch_indices = batch_df['index'].tolist()
                    sample_order.append(batch_indices)
            if sample_order_file is not None:
                with open(sample_order_file, 'wb') as file:
                    pickle.dump(sample_order, file)

        if data_list_file is not None and os.path.exists(data_list_file):
            with open(data_list_file, 'rb') as file:
                data_list = pickle.load(file)
        else:       
            data_list = []
            for batch_indices in sample_order:
                data_list.append(Batch.from_data_list(ds_ori[batch_indices]))
            if data_list_file is not None:
                with open(data_list_file, 'wb') as file:
                    pickle.dump(data_list, file)

        # data_list = data_list[:200]
        ds_batch = InMemoryDataset()
        ds_batch.data, ds_batch.slices = ds_batch.collate(data_list)

        return ds_batch


    def transform(self, data):

        data.pos = remove_mean(data.pos, dim=0).to(self.device)  # [N, 3] zero center of mass
        # data.pos = data.pos.to(self.device)  # [N, 3] zero center of mass
        data.zx = torch.randn_like(data.x).to(self.device)
        data.zcharges = torch.randn_like(data.charges).to(self.device)
        data.zpos = remove_mean(torch.randn_like(data.pos), dim=0).to(self.device)  # [N, 3] zero center of mass
        data.edge_index = self.make_adjacency_matrix(data.x.shape[0]).to(self.device)
        data.edge_attr = None
        data.recovery = torch.tensor(0).to(self.device)
        # data.z = None
        return data

    def make_adjacency_matrix(self, n_nodes):
        full_adj = self.full_adj[:, :n_nodes, :n_nodes].reshape(2, -1)
        diag_bool = self.diag_bool[:n_nodes, :n_nodes].reshape(-1)
        return full_adj[:, ~diag_bool]

    @classmethod
    def initiate_evaluation_dataloader(cls, data_num, n_node_histogram, batch_size=4, device="cpu"):
        """
        Initiate a dataloader for evaluation, which will generate data from prior distribution with n_node_histogram
        """
        max_n_nodes = len(n_node_histogram) + 10
        n_node_histogram = np.array(n_node_histogram / np.sum(n_node_histogram))
        full_adj, diag_bool = _make_global_adjacency_matrix(max_n_nodes)
        make_adjacency_matrix = lambda x: full_adj[:, :x, :x].reshape(2, -1)[
            :, ~(diag_bool[:x, :x].reshape(-1))
        ]

        def _evaluate_transform(data):
            # sample n_nodes from n_node_histogram
            n_nodes = np.random.choice(n_node_histogram.shape[0], p=n_node_histogram)
            data.zx = torch.randn(n_nodes, cls.num_atom_types).to(device)
            data.zcharges = torch.randn(n_nodes, 1).to(device)
            data.zpos = remove_mean(torch.randn(n_nodes, 3)).to(device)
            data.x = torch.randn_like(data.zpos).to(device)
            data.edge_index = make_adjacency_matrix(n_nodes).to(device)
            data.num_nodes = torch.tensor(n_nodes).to(device)
            data.recovery = torch.tensor(0).to(device)
            return data

        data_list = [Data() for _ in range(data_num // batch_size)]
        data_list = list(map(_evaluate_transform, data_list))
        data_list = [data_list[i] for i in range(data_num // batch_size) for j in range(batch_size)]
        ds = InMemoryDataset()
        ds.data, ds.slices = ds.collate(data_list)
        ds_batch = cls.build_ds_batch(cls, ds, batch_size)

        test_list = [Data() for _ in range(batch_size * 10)]
        test_list = list(map(_evaluate_transform, test_list))
        test_ds = InMemoryDataset()
        test_ds.data, test_ds.slices = test_ds.collate(test_list)
        test_ds_batch = cls.build_ds_batch(cls, test_ds, batch_size)

        return DataLoader(ds_batch, batch_size=1, shuffle=False), DataLoader(test_ds_batch, batch_size=1, shuffle=False)




