## 获得数据
Alexnet使用Imagenet数据，其中输入图片大小一般是 224×224224×224 。因为Imagenet数据训练时间过长，我们还是用前面的FashionMNIST来演示。读取数据的时候我们额外做了一步将数据扩大到原版Alexnet使用的 224×224224×224 。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
from mxnet import image

def transform(data, label):
    # resize from 28 x 28 to 224 x 224
    data = image.imresize(data, 224, 224) 
    return utils.transform_mnist(data, label)

batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size, transform)
```

##  丢弃法

```{.python .input  n=2}
from mxnet import nd

def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1
    # 这种情况下把全部元素都丢弃。
    if keep_probability == 0:
        return X.zeros_like()
    
    # 随机选择一部分该层的输出作为丢弃元素。
    mask = nd.random.uniform(
        0, 1.0, X.shape, ctx=X.context) < keep_probability
    # 保证 E[dropout(X)] == X
    scale =  1 / keep_probability 
    return mask * X * scale
```

## 定义模型
我们尝试使用GPU运行本教程代码。

```{.python .input  n=3}
ctx = utils.try_gpu()
ctx
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

先定义参数。

```{.python .input  n=4}
from mxnet import ndarray as nd 
weight_scale = .01

# output channels = 96, kernel = (11, 11)
c1 = 96
W1 = nd.random.normal(shape=(c1, 1, 11, 11), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(c1, ctx=ctx)

# batch norm 1
gamma1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(c1, ctx=ctx)
moving_variance1 = nd.zeros(c1, ctx=ctx)

# output channels = 256, kernel = (5, 5)
c2 = 256
W2 = nd.random_normal(shape=(c2, c1, 5, 5), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(c2, ctx=ctx)

# batch norm 2
gamma2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(c2, ctx=ctx)
moving_variance2 = nd.zeros(c2, ctx=ctx)

# output channels = 384, kernel = (3,3)
c3 = 384
W3 = nd.random_normal(shape=(c3, c2, 3, 3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(c3, ctx=ctx)

# batch norm 3
gamma3 = nd.random.normal(shape=c3, scale=weight_scale, ctx=ctx)
beta3 = nd.random.normal(shape=c3, scale=weight_scale, ctx=ctx)
moving_mean3 = nd.zeros(c3, ctx=ctx)
moving_variance3 = nd.zeros(c3, ctx=ctx)

# output channels = 384, kernel = (3,3)
c4 = 384
W4 = nd.random_normal(shape=(c4, c3, 3, 3), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(c4, ctx=ctx)

# batch norm 4
gamma4 = nd.random.normal(shape=c4, scale=weight_scale, ctx=ctx)
beta4 = nd.random.normal(shape=c4, scale=weight_scale, ctx=ctx)
moving_mean4 = nd.zeros(c4, ctx=ctx)
moving_variance4 = nd.zeros(c4, ctx=ctx)

# output channels = 256, kernel = (3,3)
c5 = 256
W5 = nd.random_normal(shape=(c5, c4, 3, 3), scale=weight_scale, ctx=ctx)
b5 = nd.zeros(c5, ctx=ctx)

# batch norm 5
gamma5 = nd.random.normal(shape=c5, scale=weight_scale, ctx=ctx)
beta5 = nd.random.normal(shape=c5, scale=weight_scale, ctx=ctx)
moving_mean5 = nd.zeros(c5, ctx=ctx)
moving_variance5 = nd.zeros(c5, ctx=ctx)

# output dim = 4096
o6 = 4096
W6 = nd.random.normal(shape=(112896, o6), scale=weight_scale, ctx=ctx)
b6 = nd.zeros(o6, ctx=ctx)

# output dim = 4096
o7 = 4096
W7 = nd.random.normal(shape=(4096, o7), scale=weight_scale, ctx=ctx)
b7 = nd.zeros(o7, ctx=ctx)

# output dim = 10
W8 = nd.random_normal(shape=(W7.shape[1], 10), scale=weight_scale, ctx=ctx)
b8 = nd.zeros(W8.shape[1], ctx=ctx)

# 注意这里moving_*是不需要更新的
params = [W1, b1, gamma1, beta1, 
          W2, b2, gamma2, beta2, 
          W3, b3, gamma3, beta3, 
          W4, b4, gamma4, beta4, 
          W5, b5, gamma5, beta5, 
          W6, b6, W7, b7, W8, b8]

for param in params:
    param.attach_grad()
```

下面定义模型

```{.python .input  n=5}
drop_prob6 = 0.2
drop_prob7 = 0.5

def net(X, verbose=False):
    X = X.as_in_context(W1.context)
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(3, 3), stride=(2, 2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(3, 3), stride=(2,2))
    # 第三层卷积
    h3_conv = nd.Convolution(
        data=h2, weight=W3, bias=b3, kernel=W3.shape[2:], num_filter=W3.shape[0])
    h3_activation = nd.relu(h3_conv)
    # 第四层卷积
    h4_conv = nd.Convolution(
        data=h3_activation, weight=W4, bias=b4, kernel=W4.shape[2:], num_filter=W4.shape[0])
    h4_activation = nd.relu(h4_conv)
    # 第五层卷积
    h5_conv = nd.Convolution(
        data=h4_activation, weight=W5, bias=b5, kernel=W5.shape[2:], num_filter=W5.shape[0])
    h5_activation = nd.relu(h5_conv)
    h5 = nd.Pooling(data=h5_activation, pool_type="max", kernel=(3, 3), stride=(2,2))
    
    h5 = nd.flatten(h5)
    # 第一层全连接
    h6_linear = nd.dot(h5, W6) + b6
    h6 = nd.relu(h6_linear)
    # 在第一层全连接后添加丢弃层。
    h6 = dropout(h6, drop_prob6)
    # 第二层全连接
    h7_linear = nd.dot(h6, W7) + b7
    h7 = nd.relu(h7_linear)
    # 在第二层全连接后添加丢弃层。
    h7 = dropout(h7, drop_prob7)
    # 第三层全连接
    h8_linear = nd.dot(h7, W8) + b8
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('3nd conv block:', h3_activation.shape)
        print('4nd conv block:', h4_activation.shape)
        print('5nd conv block:', h5.shape)
        print('1st dense:', h6.shape)
        print('2nd dense:', h7.shape)
        print('3rd dense:', h8_linear.shape)
        print('output:', h8_linear)
    return h8_linear
```

测试一下，输出中间结果形状（当然可以直接打印结果)和最终结果。

```{.python .input  n=6}
for data, _ in train_data:
    net(data, verbose=True)
    break
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "1st conv block: (64, 96, 106, 106)\n2nd conv block: (64, 256, 50, 50)\n3nd conv block: (64, 384, 48, 48)\n4nd conv block: (64, 384, 46, 46)\n5nd conv block: (64, 112896)\n1st dense: (64, 4096)\n2nd dense: (64, 4096)\n3rd dense: (64, 10)\noutput: \n[[ -2.15423090e-04   7.62154959e-05  -1.82979071e-04   1.76043846e-04\n    1.83784301e-04   3.45303051e-05  -1.12126836e-04   1.75724650e-04\n    5.28164892e-05  -3.08752875e-04]\n [  1.24027580e-03  -1.12966998e-04   2.41669564e-04   1.05276878e-04\n    1.08389673e-03   2.37605374e-04  -8.33832892e-04   9.09877010e-04\n    1.19367882e-03  -1.46151450e-03]\n [  1.03897427e-03  -9.18185979e-04   4.02471371e-04   1.00750825e-04\n    1.02961538e-04  -6.09696086e-04  -4.05447558e-04  -7.90179474e-04\n    3.20944644e-04  -4.89004422e-04]\n [  5.15536580e-04  -2.35673637e-04  -1.39809668e-03   1.96805748e-04\n   -5.74036909e-04   1.80147908e-04   1.22414483e-03  -5.54310216e-04\n    9.10444214e-05  -5.66272938e-04]\n [  1.05941819e-03  -3.25728091e-04   1.36962440e-03  -1.91877573e-03\n   -3.39523685e-05   1.04914652e-03   1.97776686e-03  -1.20026048e-03\n    4.71934705e-04  -1.88839994e-03]\n [  2.99800275e-04  -5.51974284e-04   1.10993488e-03   1.51627138e-03\n    5.75665385e-04   2.13537191e-04   4.97715140e-04  -5.34027000e-04\n    1.55174674e-03  -7.86341727e-04]\n [  1.46625086e-03  -9.29326401e-04   5.62372850e-04  -4.67213657e-04\n   -3.59621568e-04   8.50880286e-04  -2.82693305e-04   8.91895965e-04\n   -1.88900786e-03  -2.42378539e-03]\n [  2.33286107e-03  -1.34418777e-03   6.90428773e-04  -2.84693670e-07\n    7.91241066e-04  -4.64172248e-04   3.18049570e-04   1.22921279e-04\n    5.32803359e-04  -8.63955123e-04]\n [  3.06584523e-04  -2.36625347e-04  -9.62023405e-05  -8.38827342e-04\n    3.03782072e-05   2.21816954e-04   1.08295979e-04   8.88686773e-05\n    2.56993866e-04  -6.01637352e-04]\n [  3.63998092e-03   7.79892143e-04   6.63372106e-04   2.11248407e-05\n    1.06197561e-03  -9.97872907e-04   1.20134745e-03  -3.98157688e-04\n    1.15195173e-03  -1.24190119e-03]\n [  7.67693098e-04  -1.37410488e-03   3.43255902e-04  -4.47578059e-04\n   -7.03208148e-04  -5.60955377e-04   5.06672950e-04  -1.10816862e-03\n    4.17357835e-04  -1.59143028e-03]\n [  5.60786342e-04   1.02340709e-03  -1.30539673e-04  -4.66255529e-04\n    3.23548040e-04   5.42658847e-04  -4.61516829e-05  -1.07870577e-03\n    1.35605468e-03  -7.91651255e-04]\n [  1.02946407e-03  -2.07098085e-03  -6.86805288e-04   9.93637135e-04\n    2.69490760e-04  -4.77764697e-04  -6.18123449e-04  -3.00863583e-04\n    7.12735404e-04   2.59413995e-04]\n [  4.79343696e-04  -4.10508044e-04   9.05309163e-04   5.78066683e-04\n    1.04977970e-03  -3.33556382e-05  -5.59764914e-04  -1.19608757e-03\n    2.61937384e-06  -1.16803520e-03]\n [  8.78360588e-04  -1.07589434e-03   5.76558465e-04   6.77201082e-04\n    9.22862848e-04  -1.69229839e-04  -2.88524025e-05  -8.20624118e-04\n    1.03043986e-03  -9.23150510e-04]\n [  1.10328593e-03  -9.36089549e-04   9.27293440e-05  -5.66336443e-04\n    6.71802671e-04  -7.30789441e-04   7.49931904e-04  -6.15628727e-04\n    1.40917150e-03   1.15423973e-04]\n [  2.06598686e-03   2.30739533e-05  -1.75081845e-03   6.76289666e-04\n    6.38843747e-04  -1.05378369e-03  -3.74436873e-04  -3.06173461e-04\n    5.39923611e-04  -1.68269908e-03]\n [  4.78872302e-04   7.08140782e-04  -6.97292038e-04   1.64799334e-03\n   -6.65124506e-04   2.57308217e-04   6.08941191e-04  -3.32821888e-04\n    8.12922954e-04   2.42403679e-04]\n [  5.88391442e-04  -6.01332285e-04   4.78344329e-04  -1.29923079e-04\n    3.47423455e-04   7.24821730e-05  -1.75190304e-04  -1.12930895e-04\n    1.84971825e-04  -8.40996858e-04]\n [  1.20323279e-03  -6.07178255e-04   5.50568453e-04   5.98864688e-04\n   -1.09144440e-03   3.05670197e-04  -1.10198383e-03  -7.45068479e-04\n   -5.57912688e-04  -8.50185403e-04]\n [  2.11653998e-03  -1.40179868e-03  -8.38148815e-04   2.98241939e-05\n   -1.68065264e-04  -8.38563312e-04   4.36681701e-04   5.65562223e-04\n    1.35197770e-03   1.11934896e-04]\n [  2.22705049e-03  -1.47743348e-03   3.82369384e-04  -6.02070068e-05\n   -2.54301936e-04  -2.37249210e-03  -9.09402268e-04   4.27103165e-04\n    1.11655402e-03  -9.22674895e-04]\n [  6.00385771e-04  -6.76312600e-04   5.69913391e-06   3.42773914e-04\n   -7.18810916e-05  -7.72991218e-04  -2.20357731e-04  -2.83545378e-04\n    5.98632672e-04  -7.87107972e-04]\n [  8.37811560e-04  -2.22846196e-04   3.89075140e-04  -4.20385739e-04\n    6.46844419e-05   3.59520782e-04  -2.07150224e-05  -1.01346313e-03\n    1.18379015e-03   1.13055925e-03]\n [  9.63892147e-04  -5.91984208e-05  -1.29461114e-04  -6.17234968e-04\n    3.51014838e-04   1.32630719e-03  -1.33351889e-04   1.14240771e-04\n   -4.80056216e-04  -1.40246775e-04]\n [  7.61166797e-04   3.42211453e-04   2.08344893e-04  -1.33817480e-03\n    4.85048164e-04  -1.36735092e-04   3.11457814e-04   1.41188444e-04\n    2.79348751e-04  -1.79225369e-03]\n [  1.17189286e-03  -9.01985914e-04   3.75284522e-04  -1.85623369e-03\n    1.83501677e-03   2.82064138e-05  -1.45922205e-03   3.30516123e-05\n    8.08083569e-04  -4.75568522e-05]\n [  1.22418103e-03  -1.13348733e-03   6.35951292e-04  -3.44511791e-04\n   -3.88376793e-05  -8.27682263e-04  -4.36106056e-04  -3.90107511e-04\n    2.31243612e-05  -5.60233311e-04]\n [  2.63734219e-05   1.57453091e-04  -1.21482415e-04  -3.45726672e-04\n   -3.76492389e-05  -5.42731374e-04   3.90248635e-04  -1.01806410e-03\n    6.92328613e-05  -2.16851523e-03]\n [  1.70422322e-03  -5.12351748e-04   6.35426200e-04   5.13596751e-06\n    1.09523139e-03   3.89707508e-04   3.76677490e-04  -6.23880071e-04\n    2.53884122e-04  -6.75522489e-04]\n [  1.33310095e-03  -8.84153415e-04  -1.25524669e-03   6.41765830e-04\n   -6.47524546e-04   9.77236428e-04  -6.79899589e-04  -6.15776051e-04\n    7.50127016e-04  -2.43957256e-04]\n [  1.43166631e-03  -3.39821010e-04  -1.21693767e-03   5.09290476e-05\n    4.58678347e-04  -2.87031697e-04  -9.29562375e-04   1.47727248e-03\n    1.10703567e-03  -1.73609215e-03]\n [  1.56629470e-03   3.82564176e-04   7.12017412e-04   2.11306819e-04\n    1.01831253e-03   5.27008160e-06   9.73483897e-04  -3.56290577e-04\n    1.05751341e-03  -1.64050283e-03]\n [ -6.20168226e-04  -1.49239553e-04  -3.36476442e-05   3.70886119e-05\n    4.65729740e-04  -1.38313335e-03  -3.85704829e-04   1.00956869e-03\n    8.66869988e-04  -6.56074262e-04]\n [  1.56062725e-03  -4.59581934e-04  -1.37992145e-04   7.76561152e-04\n    1.16625964e-03  -8.24567862e-04  -6.68245542e-04  -9.08736722e-04\n    7.39951734e-04  -4.52697597e-04]\n [  1.24283799e-03  -7.45624187e-04  -6.56683231e-04   4.64725483e-04\n    8.67865048e-04  -6.33308897e-04   1.20368612e-04   6.04934001e-04\n   -2.81164801e-04   1.11619112e-04]\n [  7.39955576e-04  -2.34302614e-04  -4.77489026e-04  -7.24225538e-04\n    4.51063679e-05  -7.73181237e-05   6.29038550e-06  -7.42833479e-04\n    1.27059768e-03  -2.72748468e-04]\n [  1.11218845e-03  -9.78719443e-04   6.09989744e-04   2.19318579e-04\n    6.57071359e-04  -3.23358690e-05   1.62297249e-04  -7.53896718e-04\n    9.69057553e-04   4.73080145e-05]\n [  2.73237732e-04  -6.07898575e-04   4.15969291e-04   2.80891225e-04\n   -3.08251067e-04  -5.67090116e-04  -4.31346649e-04  -6.34508833e-05\n   -4.52076929e-05  -5.24700736e-04]\n [  9.38082288e-04   2.49145960e-04  -5.02196839e-04   1.81875599e-04\n    2.49945559e-04  -6.06763409e-04   7.58274982e-04  -4.75969282e-04\n    7.63555523e-04   2.43520859e-04]\n [  6.89023233e-04  -5.40124252e-04   2.48302909e-04  -4.59767238e-04\n    1.30455184e-04   1.51391374e-04   1.36044645e-03  -9.20000020e-04\n   -2.18202418e-04  -1.26697868e-03]\n [  9.38473328e-04  -3.02963133e-04   9.03077656e-04  -8.22265647e-05\n    6.58815901e-04  -8.01983697e-04  -6.02287822e-04  -7.09924265e-04\n    8.09851976e-04  -8.03491101e-04]\n [  7.60618248e-04   1.04232371e-04   5.49143588e-04   6.06310205e-04\n    1.11525599e-03  -1.10770925e-03   4.52571287e-04  -1.14465656e-03\n    1.90606585e-03  -2.52978830e-03]\n [  5.13958279e-04  -1.10266404e-03  -2.60439738e-05  -5.34973340e-04\n   -7.02591205e-04   3.85964318e-04  -2.72002537e-04  -6.93797367e-04\n    2.65223323e-04  -6.87430555e-04]\n [  8.05591699e-04   1.35566312e-04   4.01073426e-04  -4.30589978e-04\n   -3.59296944e-04   2.80962558e-04   7.67105303e-05  -4.53230110e-04\n   -1.25091290e-04  -1.56214705e-03]\n [  2.00573332e-03  -1.41426257e-03  -1.39665487e-03  -1.72862390e-04\n    3.21901985e-04  -1.36967101e-05  -9.17428755e-04  -1.38112879e-03\n    2.89730285e-03  -4.37918527e-04]\n [  2.49803590e-04   1.92529769e-04   4.47651808e-04  -5.81112981e-04\n    3.55113356e-04  -5.55251201e-04  -5.15623833e-04  -7.75350054e-06\n    4.76794055e-04  -4.90730978e-04]\n [  1.68313226e-03  -6.29171554e-04  -3.26518813e-04  -4.02731530e-04\n    4.19918360e-04   1.03789091e-03   3.96699354e-04   3.51930474e-04\n    3.86305706e-04   8.97117716e-05]\n [  1.60683226e-03  -5.29684476e-04  -4.64619807e-04  -7.58461771e-04\n   -3.45820066e-04  -3.16322519e-04  -9.00430314e-06  -2.94066907e-04\n    9.89164575e-04  -1.11853378e-03]\n [  1.62647269e-03  -5.56000974e-04  -4.46418067e-04  -7.50668114e-04\n    8.61016597e-05   1.14794844e-03  -2.96755868e-04   7.15267844e-04\n   -4.35458991e-04  -1.18220411e-03]\n [  2.60009459e-04  -8.45875766e-04   1.25588005e-04   1.97388558e-03\n    9.65127896e-04  -9.47946974e-04   7.78126850e-05  -4.68960061e-05\n    2.17022724e-04  -2.51240400e-03]\n [  1.12629251e-03  -6.13767770e-04  -6.22360792e-04  -2.27162367e-04\n    7.64560013e-04   8.69388692e-04   6.24188324e-05  -2.45713047e-04\n    2.41585702e-04  -9.09862283e-04]\n [  1.05237486e-04  -6.82718179e-04   2.52392201e-05  -1.60226109e-03\n   -2.00559059e-03   1.77918584e-03   1.27191632e-03  -2.81972287e-04\n    1.77303341e-03   9.43749386e-04]\n [  8.17434411e-05   4.90436330e-04   1.57101880e-04  -4.39590891e-04\n    4.60919866e-04   4.24830359e-04   2.61080044e-04   2.87707488e-04\n    1.82885356e-04  -2.83571077e-04]\n [  7.25178164e-04  -6.38398458e-04  -6.71671587e-05  -1.19607750e-04\n    4.98742214e-04   8.21260968e-04  -9.76704876e-04   5.13678475e-04\n   -2.57354171e-04  -1.41034019e-03]\n [  6.33578573e-04  -9.18495352e-04  -9.82715050e-04   3.81714257e-04\n    8.12045910e-05   7.57712114e-05   1.29737321e-03  -1.77635215e-04\n    7.07339204e-04  -2.40698268e-04]\n [  9.99741955e-04  -5.30580641e-04  -1.71166204e-04   2.26927150e-05\n   -4.11237124e-08   7.38190603e-04   9.84332990e-04  -7.70031300e-04\n    7.84171396e-04  -3.95836658e-04]\n [  1.54178683e-03  -6.83603575e-04  -1.19415962e-03  -6.40051439e-04\n    1.18199352e-03  -1.97928632e-04  -9.18190752e-04   9.04897752e-05\n   -7.13670510e-04  -8.28055374e-04]\n [  1.31113781e-03  -2.90589931e-04  -2.19995331e-04   7.17410643e-04\n    2.32829931e-04   2.89232354e-04   8.16847954e-04  -5.20957925e-04\n    1.00874924e-03  -4.76203713e-04]\n [  1.88036333e-03   3.11053067e-04  -1.71406282e-04  -2.54570274e-04\n   -1.18413554e-04  -6.68092398e-05   7.63717224e-04   5.57118110e-05\n    1.34105887e-03   4.43882600e-05]\n [  1.26026687e-03   2.38273969e-05  -1.46370847e-03   1.01307407e-03\n    1.38304880e-04   3.39374004e-04  -1.86861912e-03   8.42741400e-04\n    1.71058194e-03   2.29550235e-04]\n [  5.69007010e-04  -1.60099566e-03   6.18041668e-05  -1.04147103e-03\n    5.56496787e-04   3.96096410e-04   6.12801756e-04  -3.87976819e-04\n    1.20048434e-03  -1.51387183e-03]\n [ -1.40833290e-04  -1.21175428e-03   4.43380268e-05  -1.59814942e-03\n    1.80768402e-05   9.10695526e-04   3.17082653e-04  -1.04481261e-03\n    2.03627464e-03  -1.47532055e-03]\n [  1.23755529e-03  -1.37822481e-03   3.36429017e-04  -5.19836904e-05\n    7.64749304e-04   9.81840654e-04  -1.64314348e-03  -1.07317974e-04\n    4.76184097e-04  -2.19537690e-03]]\n<NDArray 64x10 @cpu(0)>\n"
 }
]
```

## 训练

跟前面没有什么不同的

```{.python .input}
from mxnet import autograd as autograd
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

## 总结
