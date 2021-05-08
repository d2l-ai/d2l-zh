# Parameter Servers
:label:`sec_parameterserver`

As we move from a single GPU to multiple GPUs and then to multiple servers containing multiple GPUs, possibly all spread out across multiple racks and network switches,
our algorithms for distributed and parallel training need to become much more sophisticated. Details matter since different interconnects have very different bandwidth (e.g., NVLink can offer up to 100 GB/s across 6 links in an appropriate setting, PCIe 4.0 (16-lane) offers 32 GB/s, while even high speed 100GbE Ethernet only amounts to 10 GB/s). At the same time it is unreasonable to expect that a statistical modeler be an expert in networking and systems.

The core idea of the parameter server was introduced in :cite:`Smola.Narayanamurthy.2010` in the context of distributed latent variable models. A description of the push and pull semantics then followed in :cite:`Ahmed.Aly.Gonzalez.ea.2012` and a description of the system and an open source library followed in :cite:`Li.Andersen.Park.ea.2014`. In the following we will motivate the components needed for efficiency.


## Data-Parallel Training

Let us review the data parallel training approach to distributed training. We will use this to the exclusion of all others in this section since it is significantly simpler to implement in practice. There are virtually no use cases (besides deep learning on graphs) where any other strategy for parallelism is preferred since GPUs have plenty of memory nowadays. :numref:`fig_parameterserver` describes the variant of data parallelism that we implemented in :numref:`sec_multi_gpu`. The key aspect in it is that the aggregation of gradients occurs on GPU 0 before the updated parameters are rebroadcast to all GPUs.

![Left: single GPU training. Right: a variant of multi-GPU training: (1) we compute loss and gradient, (2) all gradients are aggregated on one GPU, (3) parameter update happens and the parameters are re-distributed to all GPUs.](../img/ps.svg)
:label:`fig_parameterserver`

In retrospect, the decision to aggregate on GPU 0 seems rather ad-hoc. After all, we might just as well aggregate on the CPU. In fact, we could even decide to aggregate some of the parameters on one GPU and some others on another. Provided that the optimization algorithm supports this, there is no real reason for why we could not. For instance, if we have four parameter vectors with associated gradients $\mathbf{g}_1, \ldots, \mathbf{g}_4$ we could aggregate the gradients on one GPU for each $\mathbf{g}_i$ ($i = 1, \ldots, 4$).


This reasoning seems arbitrary and frivolous. After all, the mathematics is the same throughout. However, we are dealing with real physical hardware where different buses have different bandwidth as discussed in :numref:`sec_hardware`.
Consider a real 4-way GPU server as described in :numref:`fig_bw_hierarchy`. If it is particularly well connected, it might have a 100 GbE network card. More typical numbers are in the 1--10 GbE range with an effective bandwidth of 100 MB/s to 1 GB/s.
Since the CPUs have too few PCIe lanes to connect to all GPUs directly (e.g., consumer-grade Intel CPUs have 24 lanes) we need a [multiplexer](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). The bandwidth from the CPU on a 16x Gen3 link is 16 GB/s. This is also the speed at which *each* of the GPUs is connected to the switch. This means that it is more effective to communicate between the devices.

![A 4-way GPU server.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

For the sake of the argument let us assume that the gradients are of 160 MB. In this case it takes 30 ms to send the gradients from all 3 remaining GPUs to the fourth one (each transfer takes 10 ms = 160 MB / 16 GB/s). Adding another 30 ms to transmit the weight vectors back we arrive at a total of 60 ms.
If we send all data to the CPU we incur a penalty of 40 ms since *each* of the four GPUs needs to send the data to the CPU, yielding a total of 80 ms. Lastly assume that we are able to split the gradients into 4 parts of 40 MB each. Now we can aggregate each of the parts on a different GPU *simultaneously* since the PCIe switch offers a full-bandwidth operation between all links. Instead of 30 ms this takes 7.5 ms, yielding a total of 15 ms for a synchronization operation. In short, depending on how we synchronize parameters the same operation can take anywhere from 15 ms to 80 ms. :numref:`fig_ps_distributed` depicts the different strategies for exchanging parameters.

![Parameter synchronization strategies.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Note that we have yet another tool at our disposal when it comes to improving performance: in a deep network it takes some time to compute all gradients from the top to the bottom. We can begin synchronizing gradients for some parameter groups even while we are still busy computing them for others. See e.g., :cite:`Sergeev.Del-Balso.2018` for details on how to do this in [Horovod](https://github.com/horovod/horovod).

## Ring Synchronization

When it comes to synchronization on modern deep learning hardware we often encounter significantly bespoke network connectivity. For instance, the AWS p3.16xlarge and NVIDIA DGX-2 instances share the connectivity structure of :numref:`fig_nvlink`. Each GPU connects to a host CPU via a PCIe link which operates at best at 16 GB/s. Additionally each GPU also has 6 NVLink connections, each of which is capable of transferring 300 Gbit/s bidirectionally. This amounts to around 18 GB/s per link per direction. In short, the aggregate NVLink bandwidth is significantly higher than the PCIe bandwidth. The question is how to use it most efficiently.

![NVLink connectivity on 8  V100 GPU servers (image courtesy of NVIDIA).](../img/nvlink.svg)
:label:`fig_nvlink`

It turns out that the optimal synchronization strategy is to decompose the network into two rings and to use them to synchronize data directly :cite:`Wang.Li.Liberty.ea.2018`. :numref:`fig_nvlink_twoloop` illustrates that the network can be decomposed into one ring (1-2-3-4-5-6-7-8-1) with double NVLink bandwidth and into one (1-4-6-3-5-8-2-7-1) with regular bandwidth. Designing an efficient synchronization protocol in this case is nontrivial.

![Decomposition of the NVLink network into two rings.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`


Consider the following thought experiment: given a ring of $n$ computing nodes (or GPUs) we can send gradients from the first to the second node. There it is added to the local gradient and sent on to the third node, and so on. After $n-1$ steps the aggregate gradient can be found in the last-visited node. That is, the time to aggregate gradients grows linearly with the number of nodes. But if we do this the algorithm is quite inefficient. After all, at any time there is only one of the nodes communicating. What if we broke the gradients into $n$ chunks and started synchronizing chunk $i$ starting at node $i$?
Since each chunk is of size $1/n$ the total time is now $(n-1)/n \approx 1$. In other words, the time spent to aggregate gradients *does not grow* as we increase the size of the ring. This is quite an astonishing result. :numref:`fig_ringsync` illustrates the sequence of steps on $n=4$ nodes.

![Ring synchronization across 4 nodes. Each node starts transmitting parts of gradients to its left neighbor until the assembled gradient can be found in its right neighbor.](../img/ringsync.svg)
:label:`fig_ringsync`

If we use the same example of synchronizing 160 MB across 8 V100 GPUs we arrive at approximately $2 \cdot 160 \mathrm{MB} / (3 \cdot 18 \mathrm{GB/s}) \approx 6 \mathrm{ms}$. This is better than using the PCIe bus, even though we are now using 8 GPUs. Note that in practice these numbers are a bit worse, since deep learning frameworks often fail to assemble communication into large burst transfers. 

Note that there is a common misconception that ring synchronization is fundamentally different from other synchronization algorithms. The only difference is that the synchronization path is somewhat more elaborate when compared with a simple tree.

## Multi-Machine Training

Distributed training on multiple machines adds a further challenge: we need to communicate with servers that are only connected across a comparatively lower bandwidth fabric that can be over an order of magnitude slower in some cases. 
Synchronization across devices is tricky. After all, different machines running training code will have subtly different speed. Hence we need to *synchronize* them if we want to use synchronous distributed optimization. :numref:`fig_ps_multimachine` illustrates how distributed parallel training occurs.

1. A (different) batch of data are read on each machine, split across multiple GPUs and transferred to GPU memory. There predictions and gradients are computed on each GPU batch separately.
2. The gradients from all local GPUs are aggregated on one GPU (or parts of it are aggregated over different GPUs).
3. The gradients are sent to the CPUs.
4. The CPUs send the gradients to a central parameter server which aggregates all the gradients.
5. The aggregate gradients are then used to update the parameters and the updated parameters are broadcast back to the individual CPUs.
6. The information is sent to one (or multiple) GPUs.
7. The updated parameters are spread across all GPUs.

![Multi-machine multi-GPU distributed parallel training.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Each of these operations seems rather straightforward. And, indeed, they can be carried out efficiently *within* a single machine. Once we look at multiple machines, though, we can see that the central parameter server becomes the bottleneck. After all, the bandwidth per server is limited, hence for $m$ workers the time it takes to send all gradients to the server is $\mathcal{O}(m)$. We can break through this barrier by increasing the number of servers to $n$. At this point each server only needs to store $\mathcal{O}(1/n)$ of the parameters, hence the total time for updates and optimization becomes $\mathcal{O}(m/n)$. 
Matching both numbers yields constant scaling regardless of how many workers we are dealing with. In practice we use the *same* machines both as workers and as servers. :numref:`fig_ps_multips` illustrates the design (see also :cite:`Li.Andersen.Park.ea.2014` for details). 
In particular, ensuring that multiple machines work without unreasonable delays is nontrivial. We omit details on barriers and will only briefly touch on synchronous and asynchronous updates below.

![Top: a single parameter server is a bottleneck since its bandwidth is finite. Bottom: multiple parameter servers store parts of the parameters with aggregate bandwidth.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Key--Value Stores

Implementing the steps required for distributed multi-GPU training in practice is nontrivial. 
This is why it pays to use a common abstraction, namely that of a *key--value store* with redefined update semantics. 


Across many workers and many GPUs the computation for gradient $i$ can be defined as

$$\mathbf{g}_{i} = \sum_{k \in \text{workers}} \sum_{j \in \text{GPUs}} \mathbf{g}_{ijk},$$

where $\mathbf{g}_{ijk}$ is part of gradient $i$ split on GPU $j$ of worker $k$.
The key aspect in this operation is that it is a *commutative reduction*, that is, it turns many vectors into one and the order in which the operation is applied does not matter. This is great for our purposes since we do not (need to) have fine grained control over when which gradient is received. Besides, note that this operation is independent among different $i$.

This allows us to define the following two operations: *push*, which accumulates gradients, and *pull*, which retrieves aggregate gradients. Since we have many different sets of gradients (after all, we have many layers), we need to index the gradients with a key $i$. This similarity to key--value stores, such as the one introduced in Dynamo
:cite:`DeCandia.Hastorun.Jampani.ea.2007` is not by coincidence. They, too, satisfy many similar characteristics, in particular when it comes to distributing the parameters across multiple servers.


The push and pull operations for key-value stores are described as follows:

* **push(key, value)** sends a particular gradient (the value) from a worker to a common storage. There the value is aggregated, e.g., by summing it up.
* **pull(key, value)** retrieves an aggregate value from common storage, e.g., after combining the gradients from all workers.

By hiding all the complexity about synchronization behind a simple push and pull operation we can decouple the concerns of statistical modelers who want to be able to express optimization in simple terms and the system engineers who need to deal with the complexity inherent in distributed synchronization.

## Summary

* Synchronization needs to be highly adaptive to specific network infrastructure and connectivity within a server. This can make a significant difference to the time it takes to synchronize.
* Ring-synchronization can be optimal for p3 and DGX-2 servers. For others possibly not so much.
* A hierarchical synchronization strategy works well when adding multiple parameter servers for increased bandwidth.


## Exercises

1. Can you increase the ring synchronization even further? Hint: you can send messages in both directions.
1. Is it possible to allow asynchronous communication (while computation is still ongoing)? How does it affect performance?
1. What if we lost a server during a long-running computation? How can we design a *fault tolerance* mechanism to avoid restarting the computation fully?


[Discussions](https://discuss.d2l.ai/t/366)
