from typing import Dict, Set

from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory


class BatchInputModelFactory:
    def __init__(self, module_factories: Dict[str, BatchInputModuleFactory]):
        self.module_factories = module_factories

    def get_module_names(self) -> Set[str]:
        return set(self.module_factories.keys())

    def create(self) -> Dict[str, BatchInputModule]:
        output = {}
        for name in self.module_factories:
            output[name] = self.module_factories[name].create()
        return output

    def get_module_factory(self, module_name) -> BatchInputModuleFactory:
        return self.module_factories[module_name]