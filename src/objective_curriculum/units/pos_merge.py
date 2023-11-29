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
        to better update the ones that are rare. 
        """
        super().__init__(*args, **kwargs)

        self.gate_pacing_fn_kwargs = self.task_unit_params["optional_kwargs"]["gate_pacing_fn_kwargs"]

        # initialize the pacing function  
        self.gate_pacing_fn = get_pacing_fn(
            total_steps=self.task_num_steps,
            **self.gate_pacing_fn_kwargs
        )

    def compute_loss(
        self,
        model: Module,
        inputs: Dict[str, Tensor],
        override_input_ids: Optional[Tensor] = None,
        override_lables: Optional[Tensor] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """ 
        Override compute loss to incorporate the POS merge loss which distributes some of the
        loss to the POS tags that are similar to the masked token.
        """

        if override_input_ids is not None and override_lables is not None:
            # NOTE: This happens during inference when we are passing in a custom input
            return super().compute_loss(
                model,
                inputs,
                override_input_ids=override_input_ids,
                override_lables=override_lables,
                loss_kwargs=loss_kwargs
            )

        assert(loss_kwargs is not None and "pos_lookup" in loss_kwargs),\
            "Must provide lookup matrix for POS tags in order to use POS merge"

        assert("global_step" in loss_kwargs),\
            "Must provide global step in order to use POS merge"

        pos_lookup = loss_kwargs.pop("pos_lookup")
        global_step = loss_kwargs.pop("global_step")

        # NOTE: The POS Lookup object contains suggested label_smoothing_temps, depending on the 
        # given similarity metric.
        label_smoothing_temp = pos_lookup.label_smoothing_temp

        # Get the current gate value that specifies how much of the loss to distribute
        # to the POS merge labels and how much to the correct label
        label_gate = self.gate_pacing_fn(global_step)

        # labels for masked language modeling 
        mlm_labels = inputs['labels_mlm']
        vocab_size = pos_lookup.lookup_matrix.shape[0]

        # posmerge_labels: [batch_size, vocab_size, seq_len]
        posmerge_labels = torch.zeros(
            (mlm_labels.shape[0], vocab_size, mlm_labels.shape[1]), 
            dtype=torch.float32,
            device=self.device
        ) 

        # find target labels in mlm_labels (will be -100 for non-masked tokens)
        target_indices = (mlm_labels != -100)
        masked_tokens = mlm_labels[target_indices]

        # similarity_vals: [masked_tokens_num, vocab_size]
        similarity_vals = pos_lookup.find_similar(masked_tokens).to(self.device)
        similarity_vals[torch.arange(similarity_vals.shape[0]), masked_tokens] = 0.0

        similarity_vals = torch.exp(similarity_vals / label_smoothing_temp)
        similarity_vals = similarity_vals / torch.sum(similarity_vals, dim=1, keepdim=True)

        # if label gate is 0.9, then we distribute 90% to the MLM task and 10% to the POS merge task
        similarity_vals *= (1-label_gate)
        similarity_vals[torch.arange(similarity_vals.shape[0]), masked_tokens] = label_gate

        posmerge_labels.transpose(1, 2)[target_indices] = similarity_vals

        loss_kwargs['reduction'] = 'none'

        loss_unreduced = super().compute_loss(
            model,
            inputs,
            override_input_ids=inputs["input_ids_mlm"],
            override_lables=posmerge_labels,
            loss_kwargs=loss_kwargs,
        )

        loss_mask = (mlm_labels != -100)

        loss = loss_unreduced[loss_mask].mean()

        return loss

    def check_valid_config(self) -> None:
        """
        Checks to see if the task_unit_params contain all required params
        and keyword args to succesfully run the task unit.
        """
        assert("gate_pacing_fn_kwargs" in self.task_unit_params["optional_kwargs"]
        ), "Must provide keyword arguments for the gating pacing function to use POS merge" 

        super().check_valid_config()

