# 实战Kaggle比赛——使用Gluon预测房价和K折交叉验证

本章介绍如何使用``Gluon``来实战[Kaggle比赛](https://www.kaggle.com)。我们以[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)为例，为大家提供一整套实战中常常需要的工具，例如**K折交叉验证**。我们还以``pandas``为工具介绍如何对**真实世界**中的数据进行重要的预处理，例如：

* 处理离散数据
* 处理丢失的数据特征
* 对数据进行标准化

需要注意的是，本章仅提供一些基本实战流程供大家参考。对于数据的预处理、模型的设计和参数的选择等，我们特意只提供最基础的版本。希望大家一定要通过动手实战、仔细观察实验现象、认真分析实验结果并不断调整方法，从而得到令自己满意的结果。

这是一次宝贵的实战机会，我们相信你一定能从动手的过程中学到很多。

> Get your hands dirty。

## Kaggle中的房价预测问题

[Kaggle](https://www.kaggle.com)是一个著名的供机器学习爱好者交流的平台。为了便于提交结果，请大家注册[Kaggle](https://www.kaggle.com)账号。请注意，**目前Kaggle仅限每个账号一天以内10次提交结果的机会**。所以提交结果前务必三思。

![](../img/kaggle.png)




我们以[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)为例教大家如何实战一次Kaggle比赛。请大家在动手开始之前点击[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)了解相关信息。

![](../img/house_pricing.png)



## 读入数据

比赛数据分为训练数据集和测试数据集。两个数据集都包括每个房子的特征，例如街道类型、建造年份、房顶类型、地下室状况等特征值。这些特征值有连续的数字、离散的标签甚至是缺失值'na'。只有训练数据集包括了我们需要在测试数据集中预测的每个房子的价格。数据可以从[房价预测问题](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)中下载。

[训练数据集下载地址](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv)
[测试数据集下载地址](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv)

我们通过使用``pandas``读入数据。请确保安装了``pandas`` (``pip install pandas``)。

```{.python .input  n=2}
import pandas as pd
import numpy as np

train = pd.read_csv("../data/kaggle_house_pred_train.csv")
test = pd.read_csv("../data/kaggle_house_pred_test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))  # 截取MSSubClass到SaleCondition这些列
```

我们看看数据长什么样子。

```{.python .input  n=3}
train.head()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Id</th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>...</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n      <th>SalePrice</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>8450</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>208500</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>20</td>\n      <td>RL</td>\n      <td>80.0</td>\n      <td>9600</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>5</td>\n      <td>2007</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>181500</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>68.0</td>\n      <td>11250</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>9</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>223500</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>70</td>\n      <td>RL</td>\n      <td>60.0</td>\n      <td>9550</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2006</td>\n      <td>WD</td>\n      <td>Abnorml</td>\n      <td>140000</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>84.0</td>\n      <td>14260</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>12</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>250000</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 81 columns</p>\n</div>",
   "text/plain": "   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n0   1          60       RL         65.0     8450   Pave   NaN      Reg   \n1   2          20       RL         80.0     9600   Pave   NaN      Reg   \n2   3          60       RL         68.0    11250   Pave   NaN      IR1   \n3   4          70       RL         60.0     9550   Pave   NaN      IR1   \n4   5          60       RL         84.0    14260   Pave   NaN      IR1   \n\n  LandContour Utilities    ...     PoolArea PoolQC Fence MiscFeature MiscVal  \\\n0         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n1         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n2         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n3         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n4         Lvl    AllPub    ...            0    NaN   NaN         NaN       0   \n\n  MoSold YrSold  SaleType  SaleCondition  SalePrice  \n0      2   2008        WD         Normal     208500  \n1      5   2007        WD         Normal     181500  \n2      9   2008        WD         Normal     223500  \n3      2   2006        WD        Abnorml     140000  \n4     12   2008        WD         Normal     250000  \n\n[5 rows x 81 columns]"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=4}
test.head()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Id</th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>...</th>\n      <th>ScreenPorch</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1461</td>\n      <td>20</td>\n      <td>RH</td>\n      <td>80.0</td>\n      <td>11622</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>120</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>MnPrv</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>6</td>\n      <td>2010</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1462</td>\n      <td>20</td>\n      <td>RL</td>\n      <td>81.0</td>\n      <td>14267</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>Gar2</td>\n      <td>12500</td>\n      <td>6</td>\n      <td>2010</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1463</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>74.0</td>\n      <td>13830</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>MnPrv</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>3</td>\n      <td>2010</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1464</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>78.0</td>\n      <td>9978</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>6</td>\n      <td>2010</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1465</td>\n      <td>120</td>\n      <td>RL</td>\n      <td>43.0</td>\n      <td>5005</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>HLS</td>\n      <td>AllPub</td>\n      <td>...</td>\n      <td>144</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>1</td>\n      <td>2010</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 80 columns</p>\n</div>",
   "text/plain": "     Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n0  1461          20       RH         80.0    11622   Pave   NaN      Reg   \n1  1462          20       RL         81.0    14267   Pave   NaN      IR1   \n2  1463          60       RL         74.0    13830   Pave   NaN      IR1   \n3  1464          60       RL         78.0     9978   Pave   NaN      IR1   \n4  1465         120       RL         43.0     5005   Pave   NaN      IR1   \n\n  LandContour Utilities      ...       ScreenPorch PoolArea PoolQC  Fence  \\\n0         Lvl    AllPub      ...               120        0    NaN  MnPrv   \n1         Lvl    AllPub      ...                 0        0    NaN    NaN   \n2         Lvl    AllPub      ...                 0        0    NaN  MnPrv   \n3         Lvl    AllPub      ...                 0        0    NaN    NaN   \n4         HLS    AllPub      ...               144        0    NaN    NaN   \n\n  MiscFeature MiscVal MoSold  YrSold  SaleType  SaleCondition  \n0         NaN       0      6    2010        WD         Normal  \n1        Gar2   12500      6    2010        WD         Normal  \n2         NaN       0      3    2010        WD         Normal  \n3         NaN       0      6    2010        WD         Normal  \n4         NaN       0      1    2010        WD         Normal  \n\n[5 rows x 80 columns]"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=5}
type(train)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "pandas.core.frame.DataFrame"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

数据大小如下。

```{.python .input  n=6}
train.shape
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(1460, 81)"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=7}
test.shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(1459, 80)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
all_X.head()
```

```{.json .output n=8}
[
 {
  "data": {
   "text/html": "<div>\n<style>\n    .dataframe thead tr:only-child th {\n        text-align: right;\n    }\n\n    .dataframe thead th {\n        text-align: left;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>LotConfig</th>\n      <th>...</th>\n      <th>ScreenPorch</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>8450</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>Inside</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>20</td>\n      <td>RL</td>\n      <td>80.0</td>\n      <td>9600</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>FR2</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>5</td>\n      <td>2007</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>60</td>\n      <td>RL</td>\n      <td>68.0</td>\n      <td>11250</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>Inside</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>9</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>70</td>\n      <td>RL</td>\n      <td>60.0</td>\n      <td>9550</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>Corner</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2006</td>\n      <td>WD</td>\n      <td>Abnorml</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>60</td>\n      <td>RL</td>\n      <td>84.0</td>\n      <td>14260</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>IR1</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>FR2</td>\n      <td>...</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>12</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 79 columns</p>\n</div>",
   "text/plain": "   MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \\\n0          60       RL         65.0     8450   Pave   NaN      Reg   \n1          20       RL         80.0     9600   Pave   NaN      Reg   \n2          60       RL         68.0    11250   Pave   NaN      IR1   \n3          70       RL         60.0     9550   Pave   NaN      IR1   \n4          60       RL         84.0    14260   Pave   NaN      IR1   \n\n  LandContour Utilities LotConfig      ...       ScreenPorch PoolArea PoolQC  \\\n0         Lvl    AllPub    Inside      ...                 0        0    NaN   \n1         Lvl    AllPub       FR2      ...                 0        0    NaN   \n2         Lvl    AllPub    Inside      ...                 0        0    NaN   \n3         Lvl    AllPub    Corner      ...                 0        0    NaN   \n4         Lvl    AllPub       FR2      ...                 0        0    NaN   \n\n  Fence MiscFeature MiscVal  MoSold  YrSold  SaleType  SaleCondition  \n0   NaN         NaN       0       2    2008        WD         Normal  \n1   NaN         NaN       0       5    2007        WD         Normal  \n2   NaN         NaN       0       9    2008        WD         Normal  \n3   NaN         NaN       0       2    2006        WD        Abnorml  \n4   NaN         NaN       0      12    2008        WD         Normal  \n\n[5 rows x 79 columns]"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=9}
all_X.shape
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "(2919, 79)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=10}
all_X.dtypes
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "MSSubClass         int64\nMSZoning          object\nLotFrontage      float64\nLotArea            int64\nStreet            object\nAlley             object\nLotShape          object\nLandContour       object\nUtilities         object\nLotConfig         object\nLandSlope         object\nNeighborhood      object\nCondition1        object\nCondition2        object\nBldgType          object\nHouseStyle        object\nOverallQual        int64\nOverallCond        int64\nYearBuilt          int64\nYearRemodAdd       int64\nRoofStyle         object\nRoofMatl          object\nExterior1st       object\nExterior2nd       object\nMasVnrType        object\nMasVnrArea       float64\nExterQual         object\nExterCond         object\nFoundation        object\nBsmtQual          object\n                  ...   \nHalfBath           int64\nBedroomAbvGr       int64\nKitchenAbvGr       int64\nKitchenQual       object\nTotRmsAbvGrd       int64\nFunctional        object\nFireplaces         int64\nFireplaceQu       object\nGarageType        object\nGarageYrBlt      float64\nGarageFinish      object\nGarageCars       float64\nGarageArea       float64\nGarageQual        object\nGarageCond        object\nPavedDrive        object\nWoodDeckSF         int64\nOpenPorchSF        int64\nEnclosedPorch      int64\n3SsnPorch          int64\nScreenPorch        int64\nPoolArea           int64\nPoolQC            object\nFence             object\nMiscFeature       object\nMiscVal            int64\nMoSold             int64\nYrSold             int64\nSaleType          object\nSaleCondition     object\nLength: 79, dtype: object"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=11}
all_X.dtypes != "object"
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "MSSubClass        True\nMSZoning         False\nLotFrontage       True\nLotArea           True\nStreet           False\nAlley            False\nLotShape         False\nLandContour      False\nUtilities        False\nLotConfig        False\nLandSlope        False\nNeighborhood     False\nCondition1       False\nCondition2       False\nBldgType         False\nHouseStyle       False\nOverallQual       True\nOverallCond       True\nYearBuilt         True\nYearRemodAdd      True\nRoofStyle        False\nRoofMatl         False\nExterior1st      False\nExterior2nd      False\nMasVnrType       False\nMasVnrArea        True\nExterQual        False\nExterCond        False\nFoundation       False\nBsmtQual         False\n                 ...  \nHalfBath          True\nBedroomAbvGr      True\nKitchenAbvGr      True\nKitchenQual      False\nTotRmsAbvGrd      True\nFunctional       False\nFireplaces        True\nFireplaceQu      False\nGarageType       False\nGarageYrBlt       True\nGarageFinish     False\nGarageCars        True\nGarageArea        True\nGarageQual       False\nGarageCond       False\nPavedDrive       False\nWoodDeckSF        True\nOpenPorchSF       True\nEnclosedPorch     True\n3SsnPorch         True\nScreenPorch       True\nPoolArea          True\nPoolQC           False\nFence            False\nMiscFeature      False\nMiscVal           True\nMoSold            True\nYrSold            True\nSaleType         False\nSaleCondition    False\nLength: 79, dtype: bool"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 预处理数据

我们使用pandas对数值特征做标准化处理：

$$x_i = \frac{x_i - \mathbb{E} x_i}{\text{std}(x_i)}。$$

```{.python .input  n=12}
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
```

```{.python .input}
all_X.head()
```

现在把离散数据点转换成数值标签。

```{.python .input  n=13}
numeric_feats
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',\n       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',\n       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',\n       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',\n       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',\n       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',\n       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',\n       'MoSold', 'YrSold'],\n      dtype='object')"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
all_X = pd.get_dummies(all_X, dummy_na=True)
```

把缺失数据用本特征的平均值估计。

```{.python .input}
all_X = all_X.fillna(all_X.mean())
```

下面把数据转换一下格式。

```{.python .input}
num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()
```

## 导入NDArray格式数据

为了便于和``Gluon``交互，我们需要导入NDArray格式数据。

```{.python .input}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

X_test = nd.array(X_test)
```

我们把损失函数定义为平方误差。

```{.python .input}
square_loss = gluon.loss.L2Loss()
```

我们定义比赛中测量结果用的函数。

```{.python .input}
def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)
```

## 定义模型

我们将模型的定义放在一个函数里供多次调用。这是一个基本的线性回归模型。

```{.python .input}
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

我们定义一个训练的函数，这样在跑不同的实验时不需要重复实现相同的步骤。

```{.python .input}
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs, 
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:    
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train','test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss
```

## K折交叉验证

在[过拟合](underfit-overfit.md)中我们讲过，过度依赖训练数据集的误差来推断测试数据集的误差容易导致过拟合。事实上，当我们调参时，往往需要基于K折交叉验证。

> 在K折交叉验证中，我们把初始采样分割成$K$个子样本，一个单独的子样本被保留作为验证模型的数据，其他$K-1$个样本用来训练。

我们关心K次验证模型的测试结果的平均值和训练误差的平均值，因此我们定义K折交叉验证函数如下。

```{.python .input}
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(
            net, X_val_train, y_val_train, X_val_test, y_val_test, 
            epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k
```

### 训练模型并交叉验证

以下的模型参数都是可以调的。

```{.python .input}
k = 5
epochs = 100
verbose_epoch = 95
learning_rate = 5
weight_decay = 0.0
```

给定以上调好的参数，接下来我们训练并交叉验证我们的模型。

```{.python .input}
train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train,
                                           y_train, learning_rate, weight_decay)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %
      (k, train_loss, test_loss))
```

即便训练误差可以达到很低（调好参数之后），但是K折交叉验证上的误差可能更高。当训练误差特别低时，要观察K折交叉验证上的误差是否同时降低并小心过拟合。我们通常依赖K折交叉验证误差结果来调节参数。



## 预测并在Kaggle提交预测结果（选学）

本部分为选学内容。网络不好的同学可以通过上述K折交叉验证的方法来评测自己训练的模型。

我们首先定义预测函数。

```{.python .input}
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_train, y_train, None, None, epochs, verbose_epoch, 
          learning_rate, weight_decay)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

调好参数以后，下面我们预测并在Kaggle提交预测结果。

```{.python .input}
learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
      weight_decay)
```

执行完上述代码后，会生成一个`submission.csv`文件。这是Kaggle要求的提交格式。这时我们可以在Kaggle上把我们预测得出的结果提交并查看与测试数据集上真实房价的误差。你需要登录Kaggle网站，打开[房价预测问题地址](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)，并点击下方右侧`Submit Predictions`按钮提交。

![](../img/kaggle_submit.png)



请点击下方`Upload Submission File`选择需要提交的预测结果。然后点击下方的`Make Submission`按钮就可以查看结果啦！

![](../img/kaggle_submit2.png)

再次温馨提醒，**目前Kaggle仅限每个账号一天以内10次提交结果的机会**。所以提交结果前务必三思。

## 作业：

* 运行本教程，目前的模型在5折交叉验证上可以拿到什么样的loss？
* 如果网络条件允许，在Kaggle提交本教程的预测结果。观察一下，这个结果能在Kaggle上拿到什么样的loss？
* 通过重新设计模型、调参并对照K折交叉验证结果，新模型是否比其他小伙伴的更好？除了调参，你可能发现我们之前学过的以下内容有些帮助：
    * [多层感知机 --- 使用Gluon](mlp-gluon.md)
    * [正则化 --- 使用Gluon](reg-gluon.md)
* 如果不使用对数值特征做标准化处理能拿到什么样的loss？
* 你还有什么其他办法可以继续改进模型？小伙伴们都期待学习到你独特的富有创造力的解决方案。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1039)
