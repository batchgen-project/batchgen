


from functools import wraps
import torch
from typing import Any, Optional, TypedDict

class LossKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`int`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
    """

    num_items_in_batch: Optional[int]

def set_attribute_for_modules(module: "torch.nn.Module", key: str, value: Any):
    """
    Set a value to a module and all submodules.
    """
    setattr(module, key, value)
    for submodule in module.children():
        set_attribute_for_modules(submodule, key, value)


def del_attribute_from_modules(module: "torch.nn.Module", key: str):
    """
    Delete a value from a module and all submodules.
    """
    # because we might remove it previously in case it's a shared module, e.g. activation function
    if hasattr(module, key):
        delattr(module, key)

    for submodule in module.children():
        del_attribute_from_modules(submodule, key)

def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    use_return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        is_requested_to_return_tuple = kwargs.pop("return_dict", True) is False
        is_configured_to_return_tuple = self.config.use_return_dict is False if hasattr(self, "config") else False

        # The following allows to convert output to tuple ONLY on top level forward call,
        # while internal modules of the model will return Output objects
        # to be able to use name-based attribute access in modeling code.

        # We will check if we are on top level module, if so, turn off to tuple conversion for all
        # underling calls.
        is_top_level_module = getattr(self, "_is_top_level_module", True)
        if is_configured_to_return_tuple and is_top_level_module:
            set_attribute_for_modules(self, "_is_top_level_module", False)

        try:
            output = func(self, *args, **kwargs)
            if is_requested_to_return_tuple or (is_configured_to_return_tuple and is_top_level_module):
                output = output.to_tuple()
        finally:
            # Remove the flag after the model forward call is finished.
            if is_configured_to_return_tuple and is_top_level_module:
                del_attribute_from_modules(self, "_is_top_level_module")

        return output

    return wrapper