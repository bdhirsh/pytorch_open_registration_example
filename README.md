# pytorch_open_registration_example
Example of using pytorch's open device registration API. It covers:

(1) Writing custom kernels in C++, and registering them to the PyTorch dispatcher

(2) Providing a user API for your custom device, so users can invoke the custom code using `torch.foo(..., device="custom_device")`

(3) Registering a custom memory allocator

(4) Registering a custom device guard

This repo should be run using the latest pytorch nightly. If it fails on the latest nightly, please post an issue and I'll take a look!
