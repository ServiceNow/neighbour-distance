import typing

from collections import OrderedDict

import torch


class IntermediateFeatureModule(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        feature_layers: typing.Sequence[str],
        dummy_input_size: int = 512,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.cache: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
        self.feature_layers = feature_layers

        self.output_channels = []
        for layer_name in self.feature_layers:

            layer = _recursive_getattr(self.base_model, layer_name)
            # register the hook to cache this layer:
            self._cache_activations(layer, layer_name)
            # ensure we know the output channels of this layer:
            # self.output_channels.append(_find_output_channels(layer))

        # run a dummy input through the network:
        dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)
        dummy_output_shapes = [t.shape for t in self.forward(dummy_input).values()]
        self.output_channels = [shape[1] for shape in dummy_output_shapes]

        self.size_factors = []
        input_size = dummy_input_size
        for shape in dummy_output_shapes:
            self.size_factors.append(shape[2] / input_size)
            input_size = shape[2]

    def forward(self, x: torch.Tensor) -> typing.OrderedDict[str, torch.Tensor]:
        cache_keys = [
            f"{feature_layer}_{self.device}" for feature_layer in self.feature_layers
        ]
        if not all(layer in self.cache for layer in cache_keys):
            # run forward pass of base_model
            self.base_model(x)

        output = OrderedDict(
            [
                (feature_layer, self.cache[cache_key])
                for feature_layer, cache_key in zip(self.feature_layers, cache_keys)
            ]
        )

        # remove only this device's cached results:
        for cache_key in cache_keys:
            self.cache.pop(cache_key)

        return output

    def _cache_activations(self, layer: torch.nn.Module, layer_name: str):
        # this method comes from:
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
        def hook(model, input, output):
            device = output.device
            self.cache[f"{layer_name}_{device}"] = output

        layer.register_forward_hook(hook)

    @property
    def device(self):
        return next(self.base_model.parameters()).device


def _find_output_channels(module: torch.nn.Module) -> int:
    """Get the number of output channels of a module.

    Note: This may break if the module does not apply layers in the order in
    which they were assigned to the module (this is because of pytorch's
    dynamic computation graph).

    Parameters
    ----------
    module : torch.nn.Module

    Raises
    ------
    TypeError
        If your module does not contain any convolutional layers.

    Returns
    -------
    int
        The output channels.
    """

    output_channels = None
    # This may break if the order of module.modules() doesn't match the order
    # in which layers are applied to the data
    for layer in module.modules():
        if hasattr(layer, "out_channels"):
            output_channels = layer.out_channels

    if output_channels is None:
        raise TypeError("Your module doesn't seem to have any conv layers.")

    return output_channels


def _recursive_getattr(obj: typing.Any, name: str):
    attr_name, *rest = name.split(".", maxsplit=1)
    if not _is_integer(attr_name):
        attr = getattr(obj, attr_name)
    else:
        attr = obj[int(attr_name)]
    if len(rest) > 0:
        return _recursive_getattr(attr, rest[0])
    else:
        return attr


def _is_integer(possible_int: str):
    try:
        int(possible_int)
        return True
    except ValueError:
        return False
