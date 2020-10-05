# 样式规范

## 文本

* 章节
    * 每章开头对全章做介绍
    * 结构标题一致
        * 小结
        * 练习
        * 扫码直达讨论区
        * 参考文献（如有）
    * 引用
        * 在每节结尾处引用
* 字符串
    * 使用中文双引号
* 符号描述
    * 时刻t（不是t时刻）
	* 形状使用英文标点
        * (10, 20) 不是 （10，20）
* 人称
    * 第一人称 → 我们
    * 第二人称 → 读者、你、大家
* 工具或部件
    * Gluon, MXNet, NumPy, spaCy, ResNet-18, Fashion-MNIST, matplotlib
        * 这些都作为词，不要带重音符
    * `backward`函数
        * 不是“`backward()`函数” （不要带括号）
    * `for`循环
* 术语
    * 统一使用
        * 函数（非方法）
        * 实例（非对象）
        * 区分：超参数和参数
        * 区分：小批量随机梯度下降和随机梯度下降
        * 权重、偏差、标签
        * 模型训练、模型预测（推断）
        * 训练数据集、验证数据集、测试数据集
    * 中文优先于英文
        * 首次出现，注明原英文术语
            * 无需加粗
            * 无需加引号
    * 中英文对照统一标准
        * https://github.com/mli/gluon-tutorials-zh/blob/master/README.md

## 数学

* 数学符号样式一致
    * https://github.com/goodfeli/dlbook_notation/blob/master/notation_example.pdf
* 书本页宽限制
    * 每行长度
* 引用
    * 上式和下式
    * 以上N式，以下N式
* 公式末放英文标点
    * 逗号：,
    * 句号：.
* 赋值符号
    * \leftarrow

## 图片

* 软件
    * 使用OmniGraffle制图，以100%的大小导出pdf（infinite canvas），再使用pdf2svg转成svg
* 样式
    * 格式：
        * svg
        * png
            * export resolution: 144
    * 大小：
        * 横向：不超过400像素
        * 纵向：不超过200像素
    * 粗细：
        * StickArrow
        * 1pt
		* arrow head size: 50%
    * 字体：
        * 英文：STIXGeneral, 9pt（下标和上标：6pt）
        * 中文：PingFang SC, 9pt
	* 下标和上标中的数字和括号不要斜体
    * 颜色：
        * 非填充深蓝色（与黑相近）：
            * 5B7DAA
        * 填充蓝色（与黑对比）
            * 深：66BFFF
            * 淡：B2D9FF
* 版权
    * 不使用网络图片
* 位置
    * 两张图不可以较邻近
        * 两张图拼一下
* 引用
    * 手动引用（例如，图7.1）
* matplotlib
    * 大小
    * 分辨率

## 代码

* 使用utils.py封装多次使用函数
    * 首次出现函数，书里给出函数实现
* Python规范一致
    * PEP8
        * 二元操作符换行：操作符和后一元一起换行 (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* 将相邻赋值语句尽可能合并为同一行
	* 如 num_epochs, lr = 5, 0.1
* 变量名一致
    * num_epochs
        * 迭代周期
    * num_hiddens
        * 隐藏单元个数
    * num_inputs
        * 输入个数
    * num_outputs
        * 输出个数
    * net
        * 模型
    * lr
        * 学习率
    * acc
        * 准确率
    * 迭代中
        * 特征：X
        * 标签：y, y_hat 或 Y, Y_hat
        * for X, y in data_iter
    * 数据集：
        * 特征：features或images
        * 标签：labels
        * DataLoader实例：train_iter, test_iter, data_iter
* 注释
    * 中文
    * 句末不加句号
* 书本页宽限制
    * 每行不超过78字符
        * In [X]: 79字符不会自动换行（X = 1, ..., 9）
    	* In [XX]: 78字符不会自动换行（XX = 10, 11, ..., 99）
    * 打印结果自动换行
* imports
    * import alphabetically
    * from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
* 打印名称
    * epoch（从1开始计数）, lr, loss, train acc, time
    * 5行左右
* 打印变量
    * 代码块最后一行尽量不用print()语句，例如`x, y`而不是`print('x:', x, 'y:', y)`
* 字符串
    * 使用单引号
* 其他
    * nd.f(x) → x.nd
    * random_normal → random.normal
    * multiple imports
    * .1 → 1.0
    * 1. → 1.0
    * remove namescope

## 超链接

* 内链格式
    * [“线性回归”](linear-reg.md)一节
* 外链
    * [层](http:bla)
    * 无需暴露URL

## 英翻汉的常见问题

* 遇到不确定的地方，可以翻阅中文版第一版的处理方法（即我们需要遵照的出版标准），以及查阅人工翻译 http://www.jukuu.com/
* 建立中英文术语对照表，全书术语翻译要完全一致。
* 语法要正确（如不能缺主语、谓语）、句子要通顺（硬翻不妥就意译）、不要漏内容。
* 代码注释要翻译。注意：i) 每行不要超过78字符，注释末尾不用加句号。 ii) # 后要空一个半角字符（英文空格）。iii) 如果注释与代码同行，# 前要空两个半角字符（英文空格）。iv）保留注释中的``符号（为了表示代码部分，如变量名、函数名等）。v）注释中中文和英文之间不要空格。vi）贪婪换行：只有当一行注释抵到78字符时再换行。
* 不要新加空行（这样会另起一个自然段）。
* 术语要保留英文翻译。现在很多地方漏了英文翻译。格式：*术语*（terminology）
* 正文和代码注释均使用中文标点。例如，中文括号要用全角括号（），不是英文半角()。例外：所有表示形状的括号和逗号（逗号后紧跟半角英文空格）用英文半角，例如“(批量大小, 词数)”而不是“（批量大小，词数）”
* 英文在标题里或句首全部不要首字母大写（即便在标题的第一个词）。除非本身是首字母大写的术语
* 不要客气。“您”->“你”，去掉“请”

