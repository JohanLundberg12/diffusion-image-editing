# attr_func_registry.py
from attr_func_strategy import BlueAttrFuncStrategy


class AttrFuncRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name, strategy):
        self._registry[name] = strategy

    def get(self, name):
        return self._registry[name]


attr_func_registry = AttrFuncRegistry()
attr_func_registry.register("apply_blue", BlueAttrFuncStrategy())
