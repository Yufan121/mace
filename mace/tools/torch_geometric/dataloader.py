from collections.abc import Mapping, Sequence
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from .batch import Batch
from .data import Data
from .dataset import Dataset


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        print(f"elem: {elem}")
        
        if isinstance(elem, Sequence) and all(isinstance(item, Sequence) for item in elem):
            # Process the first element as a nested list of Data, while handling the rest as normal tensors
            first_elem = [datapoint[0] for dataframe in batch for datapoint in dataframe]
            return [first_elem] 
        
        
        if isinstance(elem, Data):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"): # recursive
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and all(isinstance(item, Data) for item in elem):
            print('Yufan modification')
            return batch
        elif isinstance(elem, Sequence) and all(isinstance(item, Sequence) for item in elem):
            print('Yufan modification, new added')
            max_length = max(len(sublist) for sublist, _, _ in batch)
            padded_batch = [
                (sublist + [None] * (max_length - len(sublist)), idx1, idx2)
                for sublist, idx1, idx2 in batch
            ]
            return padded_batch
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]
        
        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):  # Deprecated...
        return self(batch)

# class Collater:
#     def __init__(self, follow_batch, exclude_keys):
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#     def __call__(self, batch):
#         print(f"batch: {batch}")
#         if isinstance(batch[0], Sequence) and all(isinstance(item, Sequence) for item in batch[0]):
#             return batch
#         else:
#             raise TypeError(f"DataLoader found invalid type: {type(batch[0])}")
        
#     def collate(self, batch):  # Deprecated...
#         return self(batch)



class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )
