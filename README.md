# 通过MXNet/Gluon来动手学习深度学习

主页在 [https://zh.gluon.ai/](https://zh.gluon.ai/)

请使用 [https://discuss.gluon.ai](https://discuss.gluon.ai) 讨论或报告问题

## 如何贡献

所有notebook是用markdown格式存储，这样方便merge改动。jupyter可以通过notedown来直接使用markdown，[参考这里安装](./chapter00_preface/install.md#使用notedown插件来读写github源文件)

build服务器在 http://gluon-ci.mxnet.io 。这台服务器有一块Nvidia M60。

所有markdown文件需要在提交前清除output，它们会在服务器上重新执行生成结果。所以需要保证每个notebook执行不要太久，目前限制是4min。

在本地可以如下build html（需要GPU支持）

```bash
conda env update -f build/build.yml
source activate gluon_zh_docs
make html
```

生成的html会在`_build/html`。

如果没有改动notebook里面源代码，所以不想执行notebook，可以使用

```
make html DO_EVAL=0
```

但这样生成的html将不含有输出结果。
