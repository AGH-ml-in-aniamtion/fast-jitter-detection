from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DeviceConfig:
    device: str = "cuda:0"
    dtype = torch.float32
