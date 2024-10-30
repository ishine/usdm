# USDM
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Modify classes from: https://github.com/huggingface/transformers

import inspect
import torch
from torch.utils.data import Dataset, SequentialSampler, RandomSampler
from transformers import Trainer
from typing import Optional


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
class SequentialTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["position_ids", "label", "label_ids"] + self.label_names))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
class RandomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["position_ids", "label", "label_ids"] + self.label_names))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return RandomSampler(self.train_dataset)