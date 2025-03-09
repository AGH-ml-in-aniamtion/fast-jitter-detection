from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Downloadable(ABC):

    @abstractmethod
    def download(self):
        ...
