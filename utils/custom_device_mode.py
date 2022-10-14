import torch
import torch.utils.cpp_extension
from torch.overrides import TorchFunctionMode

# Load the C++ extension containing your custom kernels.
foo_module = torch.utils.cpp_extension.load(
    name="custom_device_extension",
    sources=[
        "cpp_extensions/open_registration_extension.cpp",
    ],
    extra_include_paths=["cpp_extensions"],
    extra_cflags=["-g"],
    verbose=True,
)

print('Loaded custom extension.')

# The user will globally enable the below mode when calling this API
def enable_foo_device():
    m = FooDeviceMode()
    m.__enter__()
    # If you want the mode to never be disabled, then this function shouldn't return anything.
    return m

# This is a simple TorchFunctionMode class that:
# (a) Intercepts all torch.* calls
# (b) Checks for kwargs of the form `device="foo:i"`
# (c) Turns those into custom device objects: `device=foo_module.custom_device(i)`
# (d) Forwards the call along into pytorch.
class FooDeviceMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if 'device' in kwargs and 'foo' in kwargs['device']:
            device_and_idx = kwargs['device'].split(':')
            if len(device_and_idx) == 1:
                # Case 1: No index specified
                kwargs['device'] = foo_module.custom_device()
            else:
                # Case 2: The user specified a device index.
                device_idx = int(device_and_idx[1])
                kwargs['device'] = foo_module.custom_device(device_idx)
        with torch._C.DisableTorchFunction():
            return func(*args, **kwargs)
