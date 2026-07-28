import copy
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tokenizer import normalize_token_ids


class TokenizedSFTDataset(Dataset):
    """SFT dataset from prompt_token_ids + response_token_ids.

    The prompt receives loss_mask=0 and the response receives loss_mask=1.
    This is used when the prompt must contain tokenizer-specific tokens, such as
    Qwen3's empty nothink block, without training on those prompt tokens.
    """

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        del processor
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = copy.deepcopy(list(parquet_files))
        self.tokenizer = tokenizer
        self.config = config or {}
        self.max_samples = max_samples
        self.prompt_token_key = self.config.get("prompt_token_key", "prompt_token_ids")
        self.response_token_key = self.config.get("response_token_key", "response_token_ids")
        self.pad_mode = self.config.get("pad_mode", "no_padding")
        self.truncation = self.config.get("truncation", "error")
        self.max_length = int(self.config.get("max_length", 1024))
        self.shuffle = bool(self.config.get("shuffle", False))
        self.seed = self.config.get("seed", None)
        self.cache_dir = os.path.expanduser(self.config.get("cache_dir", "~/.cache/verl/sft"))

        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)
        self._read_files()

    def _read_files(self):
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        total = len(self.dataframe)
        print(f"tokenized sft dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"selected {self.max_samples} tokenized sft samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    @staticmethod
    def _ids(value):
        return [int(x) for x in normalize_token_ids(value)]

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        prompt_ids = self._ids(row[self.prompt_token_key])
        response_ids = self._ids(row[self.response_token_key])

        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        loss_mask = torch.tensor([0] * len(prompt_ids) + [1] * len(response_ids), dtype=torch.long)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={len(input_ids)} is larger than max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
                position_ids = torch.arange(len(input_ids), dtype=torch.long)
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        if self.pad_mode == "no_padding":
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}

        if self.pad_mode not in {"right", "left_right"}:
            raise ValueError(f"Unknown pad_mode {self.pad_mode}")
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            loss_mask = torch.cat([loss_mask, torch.zeros((pad_len,), dtype=torch.long)])
            position_ids = torch.cat([position_ids, torch.zeros((pad_len,), dtype=torch.long)])
        return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
