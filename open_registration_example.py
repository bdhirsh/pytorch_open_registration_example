import torch
from utils.custom_device_mode import foo_module, enable_foo_device

# This file contains an example of how to create a custom device extension
# in PyTorch, through the dispatcher.
# It also shows what two possible user API's for custom devices look like. Either:
# (1) Expose your custom device as an object, device=my_device_obj
# (2) Ask users to enable a TorchFunctionMode, so you can use device strings: device="my_device"

# Running this file prints the following:

# (Correctly) unable to create tensor on device='bar'
# (Correctly) unable to create tensor on device='foo:2'
# Creating x on device 'foo:0'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# Creating y on device 'foo:0'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
#
# Test START
# x.device=privateuseone:0, x.is_cpu=False
# y.device=privateuseone:0, y.is_cpu=False
# Calling z = x + y
# Custom aten::add.Tensor() called!
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# z.device=privateuseone:0, z.is_cpu=False
# Calling z = z.to(device="cpu")
# Custom aten::_copy_from() called!
# z_cpu.device=cpu, z_cpu.is_cpu=True
# Calling z2 = z_cpu + z_cpu
# Test END
#
# Custom allocator's delete() called!
# Creating x on device 'foo:1'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# Creating y on device 'foo:1'
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
#
# Test START
# x.device=privateuseone:0, x.is_cpu=False
# y.device=privateuseone:0, y.is_cpu=False
# Calling z = x + y
# Custom aten::add.Tensor() called!
# Custom aten::empty.memory_format() called!
# Custom allocator's allocate() called!
# z.device=privateuseone:0, z.is_cpu=False
# Calling z = z.to(device="cpu")
# Custom aten::_copy_from() called!
# z_cpu.device=cpu, z_cpu.is_cpu=True
# Calling z2 = z_cpu + z_cpu
# Test END
#
# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!
# Custom allocator's delete() called!

def test(x, y):
    print()
    print("Test START")
    # Check that our device is correct.
    print(f'x.device={x.device}, x.is_cpu={x.is_cpu}')
    print(f'y.device={y.device}, y.is_cpu={y.is_cpu}')

    # calls out custom add kernel, registered to the dispatcher
    print('Calling z = x + y')
    z = x + y
    print(f'z.device={z.device}, z.is_cpu={z.is_cpu}')

    print('Calling z = z.to(device="cpu")')
    z_cpu = z.to(device='cpu')

    # Check that our cross-device copy correctly copied the data to cpu
    print(f'z_cpu.device={z_cpu.device}, z_cpu.is_cpu={z_cpu.is_cpu}')

    # Confirm that calling the add kernel no longer invokes our custom kernel,
    # since we're using CPU t4ensors.
    print('Calling z2 = z_cpu + z_cpu')
    z2 = z_cpu + z_cpu
    print("Test END")
    print()

# Option 1: Use a TorchFunctionMode object, that will convert `device="foo"` calls
# into our custom device objects automatically.
_holder = enable_foo_device()

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='bar')
    exit("Error: you should not be able to make a tensor on an arbitrary 'bar' device.")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='bar'")

# Show that in general, passing in a custom device string will fail.
try:
    x = torch.ones(4, 4, device='foo:2')
    exit("Error: the foo device only has two valid indices: foo:0 and foo:1")
except RuntimeError as e:
    print("(Correctly) unable to create tensor on device='foo:2'")

print("Creating x on device 'foo:0'")
x1 = torch.ones(4, 4, device='foo:0')
print("Creating y on device 'foo:0'")
y1 = torch.ones(4, 4, device='foo:0')

test(x1, y1)

# Normally you would probably want this device TorchFunction translation to happen
# permanently, but this example I want to remove it.
del _holder


# Option 2: Directly expose a custom device object
# You can pass an optional index arg, specifying which device index to use.
foo_device1 = foo_module.custom_device(1)

print("Creating x on device 'foo:1'")
x2 = torch.ones(4, 4, device=foo_device1)
print("Creating y on device 'foo:1'")
y2 = torch.ones(4, 4, device=foo_device1)

test(x2, y2)
