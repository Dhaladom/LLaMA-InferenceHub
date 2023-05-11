
from accelerate.hooks import ModelHook, AlignDevicesHook, add_hook_to_module
from accelerate.utils import find_device, send_to_device
from typing import Mapping


def send_to_device_except(data, device, non_blocking=False, skip_keys=()):
    if isinstance(data, Mapping):
        return type(data)({
                k: v if k in skip_keys else send_to_device(v, device, non_blocking)
                for k, v in data.items()
        })
    else:
        return send_to_device(data, self.input_device, non_blocking)


class AlignLogitsHook(AlignDevicesHook):
    def pre_forward(self, module, *args, **kwargs):
        if self.io_same_device:
            self.input_device = find_device([args, kwargs])

        return (
            send_to_device(args, self.execution_device),
            send_to_device_except(kwargs, self.execution_device, skip_keys=("past_key_values",)),
        )

    def post_forward(self, module, output):
        if self.io_same_device and self.input_device is not None:
            output = send_to_device_except(output, self.input_device, skip_keys=("past_key_values",))
        return output


def apply_to_model(model):
    hook = AlignLogitsHook(execution_device=model._hf_hook.execution_device, io_same_device=True)
    return add_hook_to_module(model, hook)