""" Sets up the masked language modeling base task. """

# typing imports
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module

from .mlm import MLMTask
from .registry import register_task_unit

from src.utils.pacing_fn import get_pacing_fn

@register_task_unit("pos_merge")
class POSMERGETask(MLMTask):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the POS MERGE task unit. The idea is to merge in POS tags as part of the 
        masked language modeling task by smoothing out the distribution of the masked tokens
        to better update are rare. 
        """
        super().__init__(*args, **kwargs)

        self.temp_pacing_fn_kwargs = self.task_unit_params["optional_kwargs"]["temp_pacing_fn_kwargs"]

        # initialize the pacing function  
        self.temp_pacing_fn = get_pacing_fn(
            total_steps=self.task_num_steps,
            **self.temp_pacing_fn_kwargs
        )


    def compute_loss(
        self,
        model: Module,
        inputs: Dict[str, Tensor],
        loss_kwargs: Optional[Dict[str, Any]],
        **kwargs
    ) -> Tensor:
        """ 
        Override compute loss to use 

        """

        assert(loss_kwargs is not None and "pos_lookup" in loss_kwargs),\
            "Must provide lookup matrix for POS tags in order to use POS merge"

        assert("global_step" in loss_kwargs),\
            "Must provide global step in order to use POS merge"

        pos_lookup = loss_kwargs["pos_lookup"]
        global_step = loss_kwargs["global_step"]

        # apply temperature pacing function to posmerge_labels
        temperature = self.temp_pacing_fn(global_step)


        # labels for masked language modeling 
        mlm_labels = inputs['labels_mlm']
        vocab_size = pos_lookup.lookup_matrix.shape[0]

        # posmerge_labels: [batch_size, vocab_size, seq_len]
        posmerge_labels = torch.zeros((mlm_labels.shape[0], vocab_size, mlm_labels.shape[1]), dtype=torch.float32)

        # find target labels in mlm_labels (will be -100 for non-masked tokens)
        target_indices = (mlm_labels != -100)
        masked_tokens = mlm_labels[target_indices]

        # similarity_vals: [masked_tokens_num, vocab_size]
        similarity_vals = pos_lookup.find_similar(masked_tokens)
        similarity_vals = torch.exp(similarity_vals / temperature)
        similarity_vals = similarity_vals / torch.sum(similarity_vals, dim=1, keepdim=True)

        posmerge_labels.transpose(1, 2)[target_indices] = similarity_vals

        super().compute_loss(
            model,
            inputs,
            override_input_ids=inputs["input_ids_mlm"],
            override_lables=posmerge_labels,
            **kwargs
        )

            
        def check_valid_config(self) -> None:
            """
            Checks to see if the task_unit_params contain all required params
            and keyword args to succesfully run the task unit.
            """
            assert("temp_pacing_fn_kwargs" in self.task_unit_params["optional_kwargs"]
            ), "Must provide keyword arguments for the temperature pacing function to use POS merge" 
            super().check_valid_config()

