import os
import sys

if len(sys.argv) > 1:
	framework_name = sys.argv[1]
else:
	# Assume using d2l-builder docker container
	# Here all the frameworks are installed and no CUDA support
	framework_name = None

print("*"*10, "D2L Framework Version Details", "*"*10)

if framework_name:
	# Print CUDA version
	print("nvcc --version")
	print(os.system("nvcc --version"))

if framework_name=="pytorch":
	# Print PyTorch versions
	print(f"Framework Name: {framework_name}")
	import torch; print(f"torch version: {torch.__version__}")
	import torchvision; print(f"torchvision version: {torchvision.__version__}")
	import gym; print(f"gym version: {gym.__version__}")
	import gpytorch; print(f"gpytorch version: {gpytorch.__version__}")
	import syne_tune; print(f"syne_tune version: {syne_tune.__version__}")


if framework_name=="tensorflow":
	# Print TensorFlow versions
	print(f"Framework Name: {framework_name}")
	import tensorflow; print(f"tensorflow version: {tensorflow.__version__}")
	import tensorflow_probability; print(f"tensorflow_probability version: {tensorflow_probability.__version__}")

if framework_name=="jax":
	# Print JAX versions
	print(f"Framework Name: {framework_name}")
	import jax; print(f"jax version: {jax.__version__}")
	import jaxlib; print(f"jaxlib version: {jaxlib.__version__}")
	import flax; print(f"flax version: {flax.__version__}")
	import tensorflow_datasets; print(f"tensorflow_datasets version: {tensorflow_datasets.__version__}")

if framework_name=="mxnet":
	# Print MXNet versions
	print(f"Framework Name: {framework_name}")
	import mxnet; print(f"MXNet version: {mxnet.__version__}")


# Print d2lbook version
import d2lbook; print(f"d2lbook version: {d2lbook.__version__}")

print("*"*10, "D2L Framework Version Details", "*"*10)
