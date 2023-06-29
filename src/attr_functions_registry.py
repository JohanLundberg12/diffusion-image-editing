from attr_functions import (
    AttrFunc,
    SingleColorAttrFunc,
    MultiColorAttrFunc,
    NetAttrFunc,
)
from typing import Dict, Type, Optional, Any, Union


class AttrFuncRegistry:
    """Registry to hold and retrieve different attribute function strategies."""

    def __init__(self) -> None:
        self._registry: Dict[str, Union[Type[AttrFunc], AttrFunc]] = {}

    def register(self, strategy: Union[Type[AttrFunc], AttrFunc]) -> None:
        """Register a new attribute function strategy."""
        if isinstance(strategy, type):
            strategy_name = strategy.__name__
        else:
            strategy_name = strategy.name

        self._registry[strategy_name] = strategy

    def get(self, name: str, params: Optional[Dict[str, Any]] = None) -> AttrFunc:
        """
        Retrieves an attribute function strategy by name.

        Args:
            name (str): The name of the attribute function strategy to retrieve.
            params (dict, optional): A dictionary of parameters to pass to the attribute function strategy.

        Returns:
            An instance of the attribute function strategy.
        """
        strategy_class_or_instance = self._registry.get(name)

        if strategy_class_or_instance is None:
            raise ValueError(f"No strategy registered with name: {name}")

        if isinstance(strategy_class_or_instance, type):
            return (
                strategy_class_or_instance(**params)
                if params
                else strategy_class_or_instance()
            )
        else:
            return strategy_class_or_instance

    def get_attribute_functions(self) -> list:
        return list(self._registry.keys())


def create_attr_func_registry():
    """Create and return an attribute function registry with the predefined strategies."""
    attr_func_registry = AttrFuncRegistry()
    attr_func_registry.register(SingleColorAttrFunc)
    attr_func_registry.register(MultiColorAttrFunc)
    attr_func_registry.register(NetAttrFunc)

    return attr_func_registry
