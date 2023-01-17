import torch


def move_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Helper functionality for moving a batch of data that is structured as a dictionary (possibly
    nested) onto a certain device.
    Args:
        * batch (dict): dictionary of data that needs to be moved to a device
        * device (torch.device): device to move the data to
    Returns:
        * updated_batch (dict): dictionary of data that has been moved to a device
    """

    updated_batch = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            if key not in updated_batch:
                updated_batch[key] = {}
            for sub_key, sub_val in val.items():
                if sub_val is not None:
                    updated_batch[key][sub_key] = sub_val.to(device)
        else:
            if val is not None:
                updated_batch[key] = val.to(device)
    return updated_batch
