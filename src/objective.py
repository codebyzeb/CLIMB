""" Class for defining the training objetive."""

# typing imports
import logging

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizerBase,
)

from .config import ObjectiveCurriculumParams

# TODO: Expand this class to include other objectives, and specifying customs objectives
objective_cl_logger = logging.getLogger("Objective Curriculum")


def load_objective_collator(
    curriculum: ObjectiveCurriculumParams,
    tokenizer: PreTrainedTokenizerBase,
    step: int = 0,
):
    """
    Load the data collator for the training objective. DataCollators need to either be a function
    or a callable class.

    Args:
        curriculum (ObjectiveCurriculumParams): Curriculum config object
        tokenizer (torch.Tokenizer): The tokenizer used for the model
        step (int): The current step in the curriculum
    """

    # For any given step, find the highest step in the curriculum that is equal or lower than
    # the current step
    curriculum_unit_name = max(
        [
            (curr_step, curr_name)
            for (curr_step, curr_name) in curriculum.steps.items()
            if step >= curr_step
        ],
        key=lambda x: x[0],
    )[1]

    objective_cl_logger.info(
        f"Loading objective curriculum unit: {curriculum_unit_name}"
    )

    if curriculum_unit_name == "mlm":
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=curriculum.units[curriculum_unit_name][
                "mask_probability"
            ],
        )
    elif curriculum_unit_name == "pos":
        objective_cl_logger.warning(
            "POS objective is not implemented yet - using DataCollatorForWholeWordMask instead"
        )
        return DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=curriculum.units[curriculum_unit_name][
                "mask_probability"
            ],
        )
    else:
        raise NotImplementedError(
            f"Objective {curriculum_unit_name} is not implemented"
        )