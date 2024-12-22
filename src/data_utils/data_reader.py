import os
import json
import jsonpath
from typing import Generator
from abc import ABC, abstractmethod


class BaseTextReader(ABC):
    def __init__(self, path: os.PathLike):
        self.path = path
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("abstract method not implemented!")


class SimpleTextReader(BaseTextReader):
    def __init__(self, path: os.PathLike):
        super().__init__(path)

    def __call__(self) -> Generator[str, None, None]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield line


class JsonLineReader(BaseTextReader):
    def __init__(self, path: os.PathLike, json_path_rule: str):
        super().__init__(path)
        self.json_path_rule = json_path_rule

    def __call__(self) -> Generator[str, None, None]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                data = jsonpath.jsonpath(json_data, self.json_path_rule)
                yield data[0]
