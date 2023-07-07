"""
Module for establishing a pacing function for data-driven curriculum learning.
Used by the CurriculumSampler class to determine the upper limit of the sampling difficulty.
"""

from typing import Callable


def get_pacing_fn(
    pacing_fn_name: str,
    total_steps: int,
    start_percent: float,
    end_percent: float,
    starting_difficulty: float = 0.2,
    max_difficulty: float = 1.0,
) -> Callable[[int], float]:
    """
    Modified from: https://github.com/google-research/understanding-curricula/blob/main/utils/utils.py

    Args:
        * pacing_fn_name (str): The name of the pacing function to use.
        * total_steps (int): The total number of steps in the training process.
        * start_percent (float): The percentage of steps from the total number of steps that
            have been taken before we begin increasing the data difficulty
        * end_percent (float): The percentage of steps from the total number of steps that
            have been taken after which we stop increasing the data difficulty.

        * starting_difficulty (float): The starting difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 0.2 means that initially, we sample from the
            bottom 20% difficult examples.
        * max_difficulty (float): The maximum difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 1.0 means that the maximum difficulty we
            can sample is the maximum difficulty in the dataset.

    Returns:
        * (callable): A function that takes in the current step and returns the number of
            data points to use.

    """

    assert (
        start_percent < end_percent
    ), f"For the Pacing Fn: start_percent ({start_percent}) must be less than end_percent ({end_percent})"

    step_start = start_percent * total_steps
    step_end = end_percent * total_steps

    num_steps = int(step_end - step_start)

    if pacing_fn_name == "linear":
        rate = (max_difficulty - starting_difficulty) / (num_steps)

        def _linear_function(step: int):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(rate * step_diff + starting_difficulty, max_difficulty)
            )

        return _linear_function

    elif pacing_fn_name == "quad":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 2

        def _quad_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * step_diff ** 2 + starting_difficulty, max_difficulty
                )
            )

        return _quad_function

    elif pacing_fn_name == "root":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 0.5

        def _root_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * step_diff ** 0.5 + starting_difficulty,
                    max_difficulty,
                )
            )

        return _root_function

    elif pacing_fn_name == "step":

        def _step_function(step):
            if step < step_end:
                return starting_difficulty
            else:
                return max_difficulty

        return _step_function

    elif pacing_fn_name == "exp":
        import numpy as np

        c = 10
        tilde_b = starting_difficulty
        tilde_a = num_steps
        rate = (max_difficulty - tilde_b) / (np.exp(c) - 1)
        constant = c / tilde_a

        def _exp_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return float(
                min(
                    rate * (np.exp(step_diff * constant) - 1) + tilde_b,
                    max_difficulty,
                )
            )

        return _exp_function

    elif pacing_fn_name == "log":
        import numpy as np

        c = 10
        tilde_b = starting_difficulty
        tilde_a = num_steps
        ec = np.exp(-c)
        N_b = max_difficulty - tilde_b

        def _log_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start

            return min(
                N_b * (1 + (1.0 / c) * np.log(step_diff / tilde_a + ec))
                + tilde_b,
                max_difficulty,
            )

        return _log_function

    else:
        # If no pacing function is specified, set the hardest difficulty from the beginning.
        return lambda step: 1.0
