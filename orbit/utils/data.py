from typing import Any, Tuple
import torch

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data

def process_batch_data(batch_data: Any, device: str = 'cpu') -> Tuple[Any, Any]:
    device = torch.device(device)
    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor) and batch_data[0].device != device:
            batch_data = to_device(batch_data, device)
            
        if len(batch_data) == 2:
            data, target = batch_data
        elif len(batch_data) == 1:
            data = batch_data[0]
            target = None
        else:
            data = batch_data[:-1]
            target = batch_data[-1]
    
    elif isinstance(batch_data, dict):
        first_val = next(iter(batch_data.values()), None)
        if isinstance(first_val, torch.Tensor) and first_val.device != device:
            batch_data = to_device(batch_data, device)
            
        data = batch_data
        target = None 
    
    else:
        if isinstance(batch_data, torch.Tensor) and batch_data.device != device:
            batch_data = batch_data.to(device)
        data = batch_data
        target = None
    
    return data, target