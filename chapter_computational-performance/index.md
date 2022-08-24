# 计算性能
:label:`chap_performance`

在深度学习中，数据集和模型通常都很大，导致计算量也会很大。
因此，计算的性能非常重要。
本章将集中讨论影响计算性能的主要因素：命令式编程、符号编程、
异步计算、自动并行和多GPU计算。
通过学习本章，对于前几章中实现的那些模型，可以进一步提高它们的计算性能。
例如，我们可以在不影响准确性的前提下，大大减少训练时间。

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-concise
parameterserver
```
