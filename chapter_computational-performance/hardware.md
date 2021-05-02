# 硬件
:label:`sec_hardware`

构建具有出色性能的系统需要对算法和模型有很好的了解，以捕捉问题的统计方面。同时，至少对底层硬件有一点了解也是不可或缺的。本节不能替代有关硬件和系统设计的适当课程。相反，它可以作为理解为什么某些算法比其他算法更有效以及如何实现良好吞吐量的起点。一个好的设计可以很容易地产生一个数量级的变化，而这反过来又可以在能够训练网络（例如，在一周内）和根本不能（在 3 个月内，因此错过了截止日期）之间的区别。我们首先看电脑。然后我们将放大以更仔细地查看 CPU 和 GPU。最后，我们缩小以查看服务器中心或云端中多台计算机是如何连接的。  

![Latency Numbers that every programmer should know.](../img/latencynumbers.png)
:label:`fig_latencynumbers`

不耐烦的读者可能能够用 :numref:`fig_latencynumbers` 来解决。它取自科林·斯科特的 [互动帖子]（https://people.eecs.berkeley.edu/~rcs/research/interactive_latency.html），该文章很好地概述了过去十年的进展。原来的数字来自杰夫·迪恩的 [Stanford talk from 2010](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/Stanford-DL-Nov-2010.pdf)。下面的讨论解释了这些数字的一些理由，以及它们如何指导我们设计算法。下面的讨论非常高级别和粗略。显然，它不能替代适当的课程 *，而只是为了为统计建模者提供足够的信息来做出合适的设计决策。有关计算机架构的深入概述，我们请读者参阅 :cite:`Hennessy.Patterson.2011` 或最近关于该主题的课程，例如 [Arste Asanovic] 的课程（http://inst.eecs.berkeley.edu/~cs152/sp19/）。 

## 计算机

大多数深度学习研究人员和从业人员都可以使用具有相当数量的内存、计算、某种形式的加速器（如 GPU）或其倍数的计算机。计算机由以下关键组件组成： 

* 能够执行我们提供的程序的处理器（也称为 CPU）（除了运行操作系统和许多其他内容之外），通常由 8 个或更多内核组成。
* 内存 (RAM) 用于存储和检索计算结果，例如体重矢量和激活以及训练数据。
* 以太网网络连接（有时是多个），速度从 1 Gb/s 到 100 Gb/s。在高端服务器上，可以找到更高级的互连。
* 用于将系统连接到一个或多个 GPU 的高速扩展总线 (PCIe)。服务器最多有 8 个加速器，通常以高级拓扑连接，而台式机系统则有 1 个或 2 个，具体取决于用户的预算和电源的大小。
* 耐用存储，例如磁性硬盘驱动器、固态驱动器，在许多情况下都使用 PCIe 总线连接。它可以将训练数据高效地传输到系统，并根据需要存储中间检查站。

![Connectivity of components of a computer.](../img/mobo-symbol.svg)
:label:`fig_mobo-symbol`

正如 :numref:`fig_mobo-symbol` 所示，大多数组件（网络、GPU 和存储）通过 PCIe 总线连接到 CPU。它由直接连接到 CPU 的多个通道组成。例如，AMD 的 Threadripper 3 有 64 个 PCIe 4.0 通道，每条通道都能在双向传输 16 Gbit/s 数据。内存直接连接到 CPU，总带宽高达 100 Gb/s。 

当我们在计算机上运行代码时，我们需要将数据随机播放到处理器（CPU 或 GPU），执行计算，然后将结果从处理器移回 RAM 和耐用存储。因此，为了获得良好的性能，我们需要确保这种方法无缝工作，而不会任何一个系统成为主要瓶颈。例如，如果我们无法足够快地加载图像，处理器将无法做任何工作。同样，如果我们不能足够快地将矩阵移动到 CPU（或 GPU），其处理元素将会饿死。最后，如果我们想在网络中同步多台计算机，后者不应该减慢计算速度。一种选择是将沟通和计算交织在一起。让我们更详细地看看各个组件。 

## 记忆

最基本的内存用于存储需要易于访问的数据。目前 CPU 内存通常是 [DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM) 种类型，每个模块提供 20—25 Gb/s 的带宽。每个模块都有一条 64 位宽的总线。通常使用对内存模块来允许多个通道。CPU 有 2 到 4 个内存通道，即它们的峰值内存带宽介于 4 0Gb/s 到 100 Gb/s 之间。通常每个渠道有两家银行。例如，AMD 的 Zen 3 Threadripper 有 8 个插槽。 

尽管这些数字令人印象深刻，但事实上，它们只能讲述部分故事。当我们想从内存中读取一部分时，我们首先需要告诉内存模块在哪里可以找到信息。也就是说，我们首先需要将 * 地址 * 发送到 RAM。完成此操作后，我们可以选择只读一条 64 位记录或一系列记录。后者被称为 * 突发读数 *。简而言之，将地址发送到内存并设置传输大约需要 100 ns（详细信息取决于所用内存芯片的特定时序系数），每次后续传输只需 0.2 ns。简而言之，第一次读取是后续读取的 500 倍！请注意，我们每秒可以执行高达 10,000,000 次随机读取。这表明我们尽可能避免随机内存访问，而是使用突发读取（和写入）。 

如果我们考虑到我们有多个 * 银行 *，事情就会复杂一些。每家银行可以基本上独立读取内存。这意味着两件事。一方面，只要随机读取在内存中均匀分布，有效随机读取次数最多可高 4 倍。这还意味着执行随机读取仍然是一个坏主意，因为突发读取速度也快了 4 倍。另一方面，由于内存对齐到 64 位边界，因此最好将任何数据结构与相同边界对齐。在设置适当的标志时，编译器几乎会做到这一点 [automatically](https://en.wikipedia.org/wiki/Data_structure_alignment)。鼓励好奇的读者查看关于 DRAM 的讲座，例如 [Zeshan Chishti] 的讲座 (http://web.cecs.pdx.edu/~zeshan/ece585_lec5.pdf)。 

GPU 内存受到更高的带宽要求，因为它们的处理元素比 CPU 多得多。总的来说，有两种选择可以解决这些问题。首先是使内存总线显著扩大。例如，NVIDIA 的 RTX 2080 Ti 有一条 352 位宽的总线。这允许同时传输更多信息。其次，GPU 使用特定的高性能内存。消费级设备（例如 NVIDIA 的 RTX 和 Titan 系列）通常使用 [GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM) 芯片，总带宽超过 500 Gb/s。另一种方法是使用 HBM（高带宽内存）模块。它们使用截然不同的界面，直接与专用硅片上的 GPU 连接。这使得它们非常昂贵，而且它们的使用通常仅限于高端服务器芯片，例如 NVIDIA Volta V100 系列加速器。毫不奇怪，由于前者的成本较高，GPU 内存通常比 CPU 内存小 *。出于我们的目的，他们的性能特征基本上相似，速度快得多。为了本书的目的，我们可以放心地忽略细节。它们只有在调整 GPU 内核以实现高吞吐量时才重要。 

## 存储

我们看到 RAM 的一些关键特征是 * 带宽 * 和 * 延迟 *。存储设备也是如此，只是差异可能更加极端。 

### 硬盘驱动器

*硬盘驱动器 *（HDD）已经使用了半个多世纪。简而言之，它们包含许多带头的旋转拼盘，可以放置在任何给定的轨道上读写。高端磁盘在 9 个磁盘上可容纳高达 16 TB。HDD 的主要优势之一是它们相对便宜。他们的许多缺点之一是他们典型的灾难性故障模式和相对较高的读取延迟。

要理解后者，请考虑一下 HDD 在 7,200 RPM 左右（每分钟转数）旋转的事实。如果速度快得多，由于对拼盘施加的离心力，它们就会破碎。在访问磁盘上的特定扇区时，这有一个重大缺点：我们需要等到磁盘旋转到位（我们可以移动磁盘但不能加速实际磁盘）。因此，在请求的数据可用之前，可能需要 8 毫秒以上。表达这一点的常见方法是说 HDD 可以以大约 100 个 IOP 运行（每秒输入/输出操作）。过去二十年来，这一数字基本上保持不变。更糟糕的是，增加带宽同样困难（大约为 100—200 MB/s）。毕竟，每个头都读取一条比特轨，因此比特率只能随信息密度的平方根进行缩放。因此，HDD 正在迅速降级为存档存储和非常大型数据集的低级存储。 

### 固态硬盘

固态硬盘 (SSD) 使用闪存来持久存储信息。这允许快速 * 访问存储的记录。现代固态硬盘的运行速度可达 100,000 到 500,000 IOP，即比硬盘快 3 个数量级。此外，它们的带宽可以达到 1—3Gb/s，即比 HDD 快一个数量级。这些改进听起来几乎太好了，无法实现。事实上，由于固态硬盘的设计方式，它们附带了以下警告。 

* SSD 将信息存储在块中（256 KB 或更大）。它们只能作为一个整体编写，这需要很长时间。因此，SSD 上的按位随机写入性能非常差。同样，写入数据通常需要很长时间，因为必须读取、删除区块，然后用新信息重写。到目前为止，SSD 控制器和固件已开发出算法来缓解这一尽管如此，写入速度可能会慢得多，特别是对于 QLC（四级单元）SSD。提高性能的关键是维持 * 队列 * 的操作，如果可能的话，更喜欢读取和写入大块。
* 固态硬盘中的记忆细胞耗尽相对较快（通常在几千次写入之后就已经出现了）。磨损级保护算法能够将降解扩散到许多细胞中。也就是说，不建议使用 SSD 来交换文件或大型日志文件聚合。
* 最后，带宽的大幅增加迫使计算机设计师将固态硬盘直接连接到 PCIe 总线。能够处理此问题的驱动器称为 nVMe（增强的非易失性存储器），最多可以使用 4 个 PCIe 通道。在 PCIe 4.0 上，这高达 8Gb/s。

### 云存储

云存储提供了一系列可配置的性能。也就是说，向虚拟机分配存储是动态的，无论是在数量还是在速度方面，都是由用户选择的。我们建议用户在延迟过高时（例如，在培训过程中使用许多小记录时）增加预配置的 IOP 数量。 

## 中央处理器

中央处理单元 (CPU) 是任何计算机的核心。它们由许多关键组件组成：* 能够执行机器代码的处理器内核 *，* 总线 * 连接它们（特定拓扑结构在处理器型号、代和供应商之间有显著差异），以及 *Caches*，以实现比处理器更高的带宽和更低的延迟内存访问可以通过从主内存中读取。最后，几乎所有现代 CPU 都包含 * 矢量处理单元 *，以帮助高性能线性代数和卷数，因为它们在媒体处理和机器学习中很常见。 

![Intel Skylake consumer quad-core CPU.](../img/skylake.svg)
:label:`fig_skylake`

:numref:`fig_skylake` 描述了英特尔 Skylake 消费级四核 CPU。它有一个集成的 GPU、缓存和一个连接四个核心的环形总线。以太网、WiFi、蓝牙、SSD 控制器和 USB 等外围设备可以是芯片组的一部分或直接连接 (PCIe) 至 CPU。 

### 微体系结构

每个处理器内核都由一组相当复杂的组件组成。尽管各代人和供应商之间的细节不同，但基本功能几乎是标准的。前端加载指令并尝试预测将采取哪条路径（例如，用于控制流程）。然后将指令从汇编代码解码为微指令。汇编代码通常不是处理器执行的最低级别的代码。相反，复杂的指令可能会被解码为一组更低级别的操作。然后，这些将由实际的执行核心处理。后者通常能够同时执行许多操作。例如，:numref:`fig_cortexa77` 的 ARM Cortex A77 核心能够同时执行多达 8 个操作。 

![ARM Cortex A77 Microarchitecture.](../img/a77.svg)
:label:`fig_cortexa77`

这意味着高效的程序可能能够在每个时钟周期执行多条指令，前提是它们可以独立执行。并非所有单位的创建都一样。一些专注于整数指令，而另一些则针对浮点性能进行了优化。为了提高吞吐量，处理器还可能在分支指令中同时遵循多条代码路径，然后丢弃未采用的分支的结果。这就是为什么分支预测单元（在前端）很重要，以至于只追求最有前途的途径。 

### 矢量化

深度学习非常需要计算机。因此，要使 CPU 适合机器学习，需要在一个时钟周期内执行许多操作。这是通过矢量单位实现的。它们有不同的名称 : on ARM they are called NEON, on x86 they (a recent generation) are referred to as [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) units. A common aspect is that they are able to perform SIMD (single instruction multiple data) operations. :numref:`fig_neon128` 显示了如何在 ARM 上的一个时钟周期内添加 8 个短整数。 

![128 bit NEON vectorization.](../img/neon128.svg)
:label:`fig_neon128`

根据架构的选择，此类寄存器的长度最多为 512 位，允许最多 64 对数字的组合。例如，我们可能会乘以两个数字，然后将它们添加到第三个数字，也被称为融合乘法加。英特尔 [OpenVino](https://01.org/openvinotoolkit) 使用这些功能在服务器级 CPU 上实现深度学习的可观吞吐量。但是请注意，这个数字与 GPU 能够实现的目标完全相似。例如，NVIDIA 的 RTX 2080 Ti 拥有 4,352 个 CUDA 内核，每个内核都能随时处理此类操作。 

### 缓存

考虑以下情况 : we have a modest CPU core with 4 cores as depicted in :numref:`fig_skylake`，以 2 GHz 频率运行。此外，让我们假设 IPC（每时钟指令）计数为 1，并且这些单元具有启用 256 位宽度的 AVX2。让我们进一步假设至少有一个用于 AVX2 操作的寄存器需要从内存中检索。这意味着 CPU 每时钟周期消耗 $4 \times 256 \text{ bit} = 128 \text{ bytes}$ 个数据。除非我们能够每秒向处理器传输 $2 \times 10^9 \times 128 = 256 \times 10^9$ 字节，否则处理元素将会饿死。不幸的是，这种芯片的存储器接口仅支持 20—40 Gb/s 的数据传输，即少一个数量级。修复方法是尽可能避免从内存中加载 *new* 数据，而是将其缓存在 CPU 本地。这就是缓存派上用场的地方。通常使用以下名称或概念： 

* ** 注册器 ** 严格来说不是缓存的一部分。他们帮助舞台说明。也就是说，CPU 寄存器是 CPU 可以以时钟速度访问的内存位置，而不会造成任何延迟损失。CPU 有数十个寄存器。有效地使用寄存器取决于编译器（或程序员）。例如，C 编程语言有一个 `register` 关键字。
* **L1 缓存 ** 是抵御高内存带宽要求的第一道防线。L1 缓存很小（典型大小可能为 32—64 KB），通常分为数据缓存和指令缓存。当在 L1 缓存中找到数据时，访问速度非常快。如果在那里找不到它们，则搜索将在缓存层次结构中向下进行。
* **L2 缓存 ** 是下一站。根据架构设计和处理器尺寸，它们可能是独家的。它们可能只能由给定的内核访问，也可能在多个内核之间共享。二级缓存更大（通常每核 256—512 KB），慢于 L1。此外，要访问 L2 中的内容，我们首先需要检查以意识到数据不在 L1 中，这会增加少量额外的延迟。
* **L3 缓存 ** 在多个核心之间共享，可能很大。AMD 的 Epyc 3 服务器 CPU 有高达 256 MB 的高速缓存分布在多个数字中。更典型的数字在 4-8 MB 范围内。

预测接下来需要哪些内存元素是芯片设计中的关键优化参数之一。例如，建议以 *forward* 方向遍历内存，因为大多数缓存算法都会尝试 * 向前读取 * 而不是向后读。同样，将内存访问模式保持在本地也是提高性能的好方法。 

添加缓存是一把双刃剑。一方面，他们确保处理器内核不会缺少数据。与此同时，它们增加芯片尺寸，耗用了本可用于提高处理能力的面积。此外，* 缓存未命中 * 可能会很昂贵。考虑 :numref:`fig_falsesharing` 中描述的最坏情况，* 虚假共享 *。当处理器 1 上的线程请求数据时，内存位置将缓存在处理器 0 上。要获得它，处理器 0 需要停止正在执行的操作，将信息写回主内存，然后让处理器 1 从内存中读取信息。在此操作期间，两个处理器都等与高效的单处理器实现相比，这样的代码在多个处理器上运行速度很可能更慢 *。这是为什么缓存大小（除了物理大小外）有实际限制的又一个原因。 

![False sharing (image courtesy of Intel).](../img/falsesharing.svg)
:label:`fig_falsesharing`

## GPU 和其他加速器

声称如果没有 GPU，深度学习就不会成功，这并不夸张。同样，可以合理地争辩说，由于深度学习，GPU 制造商的财富大幅度增加。硬件和算法的共同演变导致了这样一种情况，即深度学习更好或坏是更好的学习是最好的统计建模范式。因此，了解 GPU 和相关加速器（如 TPU :cite:`Jouppi.Young.Patil.ea.2017`）的具体优势是值得的。 

值得注意的是，在实践中经常作出的区别：加速器已针对训练或推理进行了优化。对于后者，我们只需要计算网络中的正向传播。反向传播不需要存储中间数据。此外，我们可能不需要非常精确的计算（FP16 或 INT8 通常就足够了）。另一方面，在训练期间，所有中间结果都需要存储才能计算梯度。此外，累积梯度需要更高的精度以避免数字下溢（或溢出）。这意味着 FP16（或与 FP32 混合精度）是最低要求。所有这些都需要更快、更大的内存（HBM2 与 GDDR6）和更大的处理能力。例如，NVIDIA 的 [Turing](https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/) T4 GPU 针对推理进行了优化，而 V100 GPU 更适合培训。 

回想一下 :numref:`fig_neon128` 中所示的矢量化。向处理器内核添加矢量单元使我们能够显著提高吞吐量。例如，在 :numref:`fig_neon128` 的示例中，我们能够同时执行 16 个操作。首先，如果我们添加的运算不仅优化了向量之间的运算，而且也优化矩阵之间的运算会怎么样？这一策略导致了张量核心（很快将涵盖）。第二，如果我们添加更多的核心怎么办？简而言之，这两种策略总结了 GPU 中的设计决策。:numref:`fig_turing_processing_block` 概述了基本的处理模块。它包含 16 个整数和 16 个浮点型单位。除此之外，两个 tensor 内核加速了与深度学习相关的少数额外操作的子集。每个流媒体多处理器由四个这样的模块组成。 

![NVIDIA Turing processing block (image courtesy of NVIDIA).](../img/turing-processing-block.png)
:width:`150px`
:label:`fig_turing_processing_block`

接下来，12 个流式多处理器被分为构成高端 TU102 处理器的图形处理群集。充足的内存通道和二级缓存补充了设置。:numref:`fig_turing` 提供了相关的细节。设计这样一个器件的原因之一是，可以根据需要添加或移除单个模块，以允许更紧凑的芯片和处理良率问题（可能无法激活故障模块）。幸运的是，在 CUDA 和框架代码层下，对于休闲的深度学习研究员来说，编程这些设备完全隐藏在一起。特别是，如果有可用资源，可以在 GPU 上同时执行多个程序。尽管如此，必须注意设备的局限性，以避免选择不适合设备内存的型号。 

![NVIDIA Turing architecture (image courtesy of NVIDIA)](../img/turing.png)
:width:`350px`
:label:`fig_turing`

值得更详细地提及的最后一个方面是 * 张量核心 *。它们是最近增加对深度学习特别有效的优化电路的趋势的一个例子。例如，TPU 为快速矩阵乘法添加了收缩压阵列 :cite:`Kung.1988`。那里的设计是为了支持极少的大型操作（第一代 TPU）。Tensor 核心在另一端。它们针对涉及 $16 \times 16$ 和 $16 \times 16$ 矩阵的小型操作进行了优化，具体取决于它们的数值精度。:numref:`fig_tensorcore` 概述了优化。 

![NVIDIA tensor cores in Turing (image courtesy of NVIDIA).](../img/tensorcore.jpg)
:width:`400px`
:label:`fig_tensorcore`

显然，在优化计算时，我们最终会作出某些妥协。其中之一是 GPU 不擅长处理中断和稀疏数据。尽管存在明显的例外，例如 [Gunrock](https://github.com/gunrock/gunrock) :cite:`Wang.Davidson.Pan.ea.2016`，稀疏矩阵和向量的访问模式与 GPU 出色的高带宽突发读取操作不太好。匹配这两个目标是积极研究的一个领域。例如，请参阅 [DGL](http://dgl.ai)，这是一个专为图表进行深度学习而调整的图书馆。 

## 网络和公共汽车

每当单个设备不足以进行优化时，我们都需要将数据传入和传出该设备来同步处理。这是网络和公共汽车派上用场的地方。我们有许多设计参数：带宽、成本、距离和灵活性。一方面，我们的 WiFi 范围相当不错，非常容易使用（毕竟没有电线），价格便宜，但它提供了相对平庸的带宽和延迟。没有任何机器学习研究人员都不会用它来构建服务器群集。在接下来的内容中，我们重点介绍了适合深度学习的互连。 

* **PCIe** 是专用总线，用于每条通道的极高带宽点对点连接（在 PCIe 4.0 上，16 通道插槽中的 PCIe 4.0 最高可达 32 Gb/s）。延迟的顺序为个位数微秒（5 μs）。PCIe 链接很宝贵。处理器的数量有限：AMD 的 EPYC 3 有 128 个通道，英特尔的至强每芯片最多有 48 条通道；在台式机级 CPU 上，数字分别为 20（锐龙 9）和 16 条（酷睿 i9）。由于 GPU 通常有 16 条通道，因此这限制了能够以全带宽连接到 CPU 的 GPU 的数量。毕竟，他们需要与存储和以太网等其他高带宽外围设备共享链路。就像 RAM 访问一样，由于数据包开销降低，大批量传输更为可取。
* ** Ethernet** 是连接计算机的最常用方式。虽然它比 PCIe 慢得多，但它的安装非常便宜且有弹性，并且覆盖的距离要长得多。低级服务器的典型带宽为 1 Gbit/s。高端设备（例如云中的 [C5 instances](https://aws.amazon.com/ec2/instance-types/c5/)）提供 10 到 100 Gbit/s 的带宽。与以前的所有情况一样，数据传输都有巨大的间接费请注意，我们几乎从来不直接使用原始以太网，而是在物理互连之上执行的协议（例如 UDP 或 TCP/IP）。这进一步增加了开销。像 PCIe 一样，以太网设计用于连接两台设备，例如计算机和交换机。
* **Switch ** 允许我们以任何一对设备同时进行（通常为满带宽）点对点连接的方式连接多台设备。例如，以太网交换机可能会以较高的横截面带宽连接 40 台服务器。请注意，交换机并不是传统计算机网络所独有的。即使是 PCIe 车道也可以是 [switched](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches)。例如，将大量 GPU 连接到主机处理器时，就会发生这种情况，就像 [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/) 那样。
* **nvlink** 在非常高带宽的互连方面是 PCIe 的替代方案。它每个链路提供高达 300 Gbit/s 的数据传输速率。服务器 GPU (Volta V100) 有六个链路，而消费级 GPU (RTX 2080 Ti) 只有一条链路，以降低 100 Gbit/s 的速率运行。我们建议使用 [NCCL](https://github.com/NVIDIA/nccl) 来实现 GPU 之间的高数据传输。

## 更多延迟数字

:numref:`table_latency_numbers` 和 :numref:`table_latency_numbers_tesla` 中的摘要来自 [Eliot Eshelman](https://gist.github.com/eshelman)，他将这些数字的更新版本维持为 [GitHub gist](https://gist.github.com/eshelman/343a1c46cb3fba142c1afdcdeec17646)。 

: 常见延迟数字。 

| Action | Time | Notes |
| :----------------------------------------- | -----: | :---------------------------------------------- |
| L1 cache reference/hit                     | 1.5 ns | 4 cycles                                        |
| Floating-point add/mult/FMA                | 1.5 ns | 4 cycles                                        |
| L2 cache reference/hit                     |   5 ns | 12 ~ 17 cycles                                  |
| Branch mispredict                          |   6 ns | 15 ~ 20 cycles                                  |
| L3 cache hit (unshared cache)              |  16 ns | 42 cycles                                       |
| L3 cache hit (shared in another core)      |  25 ns | 65 cycles                                       |
| Mutex lock/unlock                          |  25 ns |                                                 |
| L3 cache hit (modified in another core)    |  29 ns | 75 cycles                                       |
| L3 cache hit (on a remote CPU socket)      |  40 ns | 100 ~ 300 cycles (40 ~ 116 ns)                  |
| QPI hop to a another CPU (per hop)         |  40 ns |                                                 |
| 64MB memory ref. (local CPU)          |  46 ns | TinyMemBench on Broadwell E5-2690v4             |
| 64MB memory ref. (remote CPU)         |  70 ns | TinyMemBench on Broadwell E5-2690v4             |
| 256MB memory ref. (local CPU)         |  75 ns | TinyMemBench on Broadwell E5-2690v4             |
| Intel Optane random write                  |  94 ns | UCSD Non-Volatile Systems Lab                   |
| 256MB memory ref. (remote CPU)        | 120 ns | TinyMemBench on Broadwell E5-2690v4             |
| Intel Optane random read                   | 305 ns | UCSD Non-Volatile Systems Lab                   |
| Send 4KB over 100 Gbps HPC fabric          |   1 μs | MVAPICH2 over Intel Omni-Path                   |
| Compress 1KB with Google Snappy            |   3 μs |                                                 |
| Send 4KB over 10 Gbps ethernet             |  10 μs |                                                 |
| Write 4KB randomly to NVMe SSD             |  30 μs | DC P3608 NVMe SSD (QOS 99% is 500μs)            |
| Transfer 1MB to/from NVLink GPU            |  30 μs | ~33GB/s on NVIDIA 40GB NVLink                 |
| Transfer 1MB to/from PCI-E GPU             |  80 μs | ~12GB/s on PCIe 3.0 x16 link                  |
| Read 4KB randomly from NVMe SSD            | 120 μs | DC P3608 NVMe SSD (QOS 99%)                     |
| Read 1MB sequentially from NVMe SSD        | 208 μs | ~4.8GB/s DC P3608 NVMe SSD                    |
| Write 4KB randomly to SATA SSD             | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                   |
| Read 4KB randomly from SATA SSD            | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                   |
| Round trip within same datacenter          | 500 μs | One-way ping is ~250μs                          |
| Read 1MB sequentially from SATA SSD        |   2 ms | ~550MB/s DC S3510 SATA SSD                    |
| Read 1MB sequentially from disk            |   5 ms | ~200MB/s server HDD                           |
| Random Disk Access (seek+rotation)         |  10 ms |                                                 |
| Send packet CA->Netherlands->CA            | 150 ms |                                                 |
:label:`table_latency_numbers`

: NVIDIA Tesla GPU 的延迟数字。 

| Action | Time | Notes |
| :------------------------------ | -----: | :---------------------------------------- |
| GPU Shared Memory access        |  30 ns | 30~90 cycles (bank conflicts add latency) |
| GPU Global Memory access        | 200 ns | 200~800 cycles                            |
| Launch CUDA kernel on GPU       |  10 μs | Host CPU instructs GPU to start kernel    |
| Transfer 1MB to/from NVLink GPU |  30 μs | ~33GB/s on NVIDIA 40GB NVLink           |
| Transfer 1MB to/from PCI-E GPU  |  80 μs | ~12GB/s on PCI-Express x16 link         |
:label:`table_latency_numbers_tesla`

## 摘要

* 设备有操作开销。因此，重要的是要瞄准少量大量转账，而不是许多小转账。这适用于 RAM、SSD、网络和 GPU。
* 矢量化是性能的关键。确保你知道加速器的具体能力。例如，一些英特尔至强 CPU 对 INT8 操作特别有用，NVIDIA Volta GPU 在 FP16 矩阵矩阵操作中表现出色，NVIDIA Timon 在 FP16、INT8 和 INT4 操作中出色。
* 由于数据类型较小而导致的数值溢出可能是训练期间的问题（以及在推理期间较小程度上）。
* 别名可能会显著降低性能。例如，64 位 CPU 上的内存对齐应该针对 64 位边界进行。在 GPU 上，保持卷积大小对齐是个好主意，例如，与张量核心保持一致。
* 将算法与硬件相匹配（例如，内存占用量和带宽）。将参数调整到缓存中时，可以实现极大的加速（数量级）。
* 我们建议您在验证实验结果之前先在纸上勾画出一种新颖算法的性能。数量级或更多的差异是令人担忧的原因。
* 使用分析器调试性能瓶颈。
* 培训和推理硬件在价格和性能方面有不同的甜点。

## 练习

1. 编写 C 代码来测试访问相对于外部存储器接口对齐或未对齐内存之间的速度是否存在任何差异。提示：小心缓存效果。
1. 测试按顺序访问内存或按给定步长访问内存之间的速度差异。
1. 你怎么能测量 CPU 上的缓存大小？
1. 您将如何在多个内存通道之间布局数据以获得最大带宽？如果你有很多小线程你会怎么布局？
1. 企业级硬盘以 10,000 rpm 的速度旋转。HDD 在读取数据之前花最坏情况的绝对最短时间是多少（你可以假设头部几乎瞬间移动）？为什么 2.5 英寸硬盘在商业服务器中变得流行（相对于 3.5 英寸和 5.25 英寸驱动器）？
1. 假设硬盘制造商将存储密度从每平方英寸 1 Tbit 增加到每平方英寸 5 Tbit。你可以在 2.5 英寸硬盘上的戒指上存储多少信息？内部和外部轨道之间有区别吗？
1. 从 8 位到 16 位数据类型将芯片的数量增加大约四倍。为什么？为什么 NVIDIA 会将 INT4 操作添加到他们的图灵 GPU 中？
1. 与向后读取相比，通过内存向前读取的速度要快多少？不同计算机和 CPU 供应商之间该数字是否有所不同？为什么？编写 C 代码然后进行试验。
1. 你能测量磁盘的缓存大小吗？典型的硬盘是什么？SSD 需要缓存吗？
1. 测量通过以太网发送消息时的数据包开销。查找 UDP 和 TCP/IP 连接之间的区别。
1. 直接内存访问允许 CPU 以外的设备直接向（从）内存中写入（和读取）。为什么这是个好主意？
1. 看看图灵 T4 GPU 的性能数字。为什么从 FP16 到 INT8 和 INT4 时，性能 “仅” 翻了一番？
1. 旧金山和阿姆斯特丹之间的往返旅行应该最短的时间是多少？提示：你可以假设距离是 10,000 公里。

[Discussions](https://discuss.d2l.ai/t/363)
