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
    start_temp: float = 10, 
    end_temp: float = 0.1,
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

        * start_temp (float): The initial temperature value used for labeling smoothing; should 
            start as a value > 1 and decrease to belowe 1 as the training progresses. 
        * end_temp (float):  The final temperature value used for labeling smoothing; should
            end up being a value < 1.
    Returns:
        * (callable): A function that takes in the current step and returns the number of
            data points to use.

    """

    assert (
        start_percent < end_percent
    ), f"For the Pacing Fn: start_percent ({start_percent}) must be less than end_percent ({end_percent})"

    assert( 
        start_temp > end_temp
    ), f"For the Pacing Fn: start_temp ({start_temp}) must be greater than end_temp ({end_temp})"
    
    step_start = start_percent * total_steps
    step_end = end_percent * total_steps

    num_steps = int(step_end - step_start)

    if pacing_fn_name == "linear":
        rate = (end_temp - start_temp) / (num_steps)

        def _linear_function(step: int):
            if step < step_start:
                return start_temp

            step_diff = step - step_start

            return float(
                max(rate * step_diff + start_temp, end_temp)
            )

        return _linear_function

    elif pacing_fn_name == "quad":
        rate = (end_temp - start_temp) / (num_steps) ** 2

        def _quad_function(step):
            if step < step_start:
                return start_temp

            step_diff = step - step_start

            return float(
                max(
                    rate * step_diff ** 2 + start_temp, end_temp
                )
            )

        return _quad_function

    elif pacing_fn_name == "root":
        rate = (end_temp - start_temp) / (num_steps) ** 0.5

        def _root_function(step):
            if step < step_start:
                return start_temp

            step_diff = step - step_start

            return float(
                max(
                    rate * step_diff ** 0.5 + start_temp,
                    end_temp,
                )
            )

        return _root_function

    elif pacing_fn_name == "step":

        def _step_function(step):
            if step < step_end:
                return start_temp
            else:
                return end_temp

        return _step_function

    elif pacing_fn_name == "exp":
        import numpy as np

        c = 10
        tilde_b = start_temp
        tilde_a = num_steps
        rate = (end_temp - tilde_b) / (np.exp(c) - 1)
        constant = c / tilde_a

        def _exp_function(step):
            if step < step_start:
                return start_temp

            step_diff = step - step_start

            return float(
                max(
                    rate * (np.exp(step_diff * constant) - 1) + tilde_b,
                    end_temp,
                )
            )

        return _exp_function

    elif pacing_fn_name == "log":
        import numpy as np

        c = 10
        tilde_b = start_temp
        tilde_a = num_steps
        ec = np.exp(-c)
        N_b = end_temp - tilde_b

        def _log_function(step):
            if step < step_start:
                return start_temp

            step_diff = step - step_start

            return max(
                N_b * (1 + (1.0 / c) * np.log(step_diff / tilde_a + ec))
                + tilde_b,
                end_temp,
            )

        return _log_function

    else:
        # If no pacing function is specified, set the hardest difficulty from the beginning.
        return lambda step: 1.0