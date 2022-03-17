import torch
import numpy as np

# x = torch.empty(2, 2, 2, 3)
# print(x)

# x = torch.rand(2, 2)
# print(x)

# x = torch.zeros(2, 2)
# print(x)

# x = torch.ones(2, 2, dtype=torch.double)
# print(x.dtype)

# x = torch.ones(2, 2, dtype=torch.float16)
# print(x.size())

# x = torch.tensor([2.5, 0.1])
# print(x)

# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# print(x)
# print(y)
# z = x + y   # or z = torch.add(x, y)
# print(z)
# z = x - y   # or z = torch.sub(x, y) 
# print(z)
# z = x * y   # or z = torch.mul(x, y)
# print(z)
# z = x / y   # or z = torch.div(x, y)
# print(z)

# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# y.add_(x)   # this will modify y and add all of the element of x to y
# print(y)

# x = torch.rand(5, 3)
# print(x)
# print(x[:, 0])          # print by column 0
# print(x[0, :])          # print by row 0
# print(x[2, 2])          # print the element x[2, 2]
# print(x[2, 2].item())   # print the value of element x[2, 2] 

# x = torch.rand(4, 4)
# print(x)
# y = x.view(16)            # print in one line
# print(y) 
# y = x.view(-1, 8)         # print in two lines with 8 values in each line
# print(y)
# print(y.size())

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(type(b))

# a.add_(1)
# print(a)
# print(b)
# -> Caution: If the tensor is on the CPU and not the GPU 
# then both objects will share the same memory location
# so this mean that if we change one we will also change 
# the other 

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(a)
# print(b)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device = device)    # Create a tensor on the GPU
#     y = torch.ones(5)                       
#     y = y.to(device)                      # Move it to my device (my GPU)
#     z = x + y                             # This will be perform on the GPU and might be much faster
#     z.to("cpu")                           # Now it would be on the CPU again

# Numpy can only handle CPU tensors, so you cannot convert a GPU tensor
# back to numpy so you must be move it back to the CPU

x = torch.ones(5, requires_grad=True)
print(x)



