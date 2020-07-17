# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

# Defined in file: ./chapter_preliminaries/pandas.md
def mkdir_if_not_exist(path):  #@save
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


# Alias defined in config.ini
size = lambda a: a.size
transpose = lambda a: a.T

ones = np.ones
zeros = np.zeros
arange = np.arange
meshgrid = np.meshgrid
sin = np.sin
sinh = np.sinh
cos = np.cos
cosh = np.cosh
tanh = np.tanh
linspace = np.linspace
exp = np.exp
log = np.log
tensor = np.array
normal = np.random.normal
matmul = np.dot
int32 = np.int32
float32 = np.float32
concat = np.concatenate
stack = np.stack
abs = np.abs
numpy = lambda x, *args, **kwargs: x.asnumpy(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.as_in_context(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.astype(*args, **kwargs)

