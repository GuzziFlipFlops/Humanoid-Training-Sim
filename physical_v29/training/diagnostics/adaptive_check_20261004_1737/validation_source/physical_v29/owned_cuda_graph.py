"""Bounded CUDA replay with copied inputs and independently owned outputs.

Only pure tensor calculations belong here: no physics step, RNG, host reads,
reset bookkeeping, observation histories, parameter updates or logging.
Graphs bind static addresses; each call copies the *current* pose into them.
"""
from collections import OrderedDict
import torch


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    raise TypeError(type(value))


class OwnedCudaCall:
    def __init__(self, function, inputs):
        self.inputs = tuple(x.detach().clone().contiguous() for x in inputs)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                function(*self.inputs)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
            self.outputs = function(*self.inputs)
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)

    def __call__(self, inputs):
        for target, current in zip(self.inputs, inputs):
            target.copy_(current)
        self.graph.replay()
        # Never expose static graph outputs: the next replay overwrites them.
        return _clone_tree(self.outputs)


class OwnedCudaCache:
    def __init__(self, max_entries=2):
        self.entries = OrderedDict()
        self.max_entries = max_entries

    def __len__(self):
        return len(self.entries)

    def call(self, function, *inputs, mode=()):
        if (not inputs or not inputs[0].is_cuda or any(x.requires_grad for x in inputs)
                or torch.cuda.is_current_stream_capturing()):
            return function(*inputs)
        # No tensor objects, storage addresses or pose values in cache keys.
        key = (mode, tuple((str(x.device), x.dtype, tuple(x.shape)) for x in inputs))
        if key not in self.entries:
            if len(self.entries) >= self.max_entries:
                # Ensure the evicted graph has no in-flight replay before releasing storage.
                torch.cuda.synchronize(inputs[0].device)
                self.entries.popitem(last=False)
            self.entries[key] = OwnedCudaCall(function, inputs)
        self.entries.move_to_end(key)
        return self.entries[key](inputs)
