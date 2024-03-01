# index 选择指南

选择 index 并不容易，所以这里有几个基本问题可以帮助选择 index。它们主要适用于L2距离。

- 我们使用 `index_factory` 字符串来表示它们（index）
- 如果有参数，我们指示它们为相应的 ParameterSpace 参数。

## 搜索次数少吗？

如果您计划只执行几次搜索（例如1000-10000），index 构建时间将不会在搜索的时候被摊销。那么直接计算是最有效的选择。

"Flat" index 可以完成这个。如果内存中放不下整个数据集，您可以一个接一个地构建小索引，然后合并搜索结果，详见[这里](https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index#combining-the-results-from-several-searches)

## 需要准确的结果吗？

### 如果是的话，请用 "Flat"

唯一能保证准确结果的 index 是 `IndexFlatL2` 或 `IndexFlatIP`。它们也为其他 index 的索引结果提供基线标准。它不会压缩 vector，也不会增加额外的开销。它不支持 `add_with_ids`，即仅支持顺序添加，因此如果您需要 `add_with_ids`，请用 `"IDMap,Flat"`. flat index 不需要训练，也没有参数。

> 支持 GPU

## 内存够用吗

请记住，所有 Faiss index 都存储在内存中。以下针对的是不精确的场景（内存受限），我们在精度和速度之间权衡并做了优化。

### 你内存很大：`HNSW`M或`IVF1024,PQ`N`x4fs,RFlat`

如果您有很多内存或数据集很小，HNSW 是最好的选择，它是一个非常快速和准确的 index。4 <= M <= 64是每个 vector 的 link 数量，越高越准确，但使用更多的内存。速度-准确性权衡指标是通过 `efSearch` 参数设置的。每个 vector 的内存使用量是 $(d * 4 + M * 2 * 4)$。

HNSW 只支持顺序添加（不支持 `add_with_ids`），所以如果需要 `add_with_ids` 请使用带 IDMap 前缀的。HNSW 不需要训练，也不支持从 index 中删除 vector。

第二个选项比 HNSW 更快。然而，它需要一个重新排名的过程，因此有两个参数需要设置：re-ranking 的`k_factor` 和 IVF 的 `nprobe`。

> 不支持 GPU

### 如果你内存一般般，不是要多少有多少，那么 `"...,Flat"`

"..." 是指数据集的聚类操作（见下文）。聚类后，`"Flat"` 只是将 vector 加载到对应的 buckets 中（不会压缩它们），因此索引存储大小与原始数据大小相同。速度和精度之间的权衡是通过 `nprobe` 参数确定的。

> 支持 GPU（前提是选的聚类方式也支持 GPU）

### 如果你有些在意内存，那么 `OPQ`M`_`D`,...,PQ`M`x4fsr`

如果存储整个向量代价比较大，那么做如下两步：
- 使用 OPQ 变化，将维度减少为 D
- 将 vector 进行 PQ 乘积量化，把每个向量元素压缩成 4-bits 大小（即一共就 M*4 大小）

显然，每个 vector 是 M/2 字节大小。

> 支持 GPU

### 如果你非常非常在意内存，那么 `OPQ`M`_`D`,...,PQ`M

`PQ`M 使用一个 product quantizer 将 vector 压缩成 M 字节大小，M 通常<=64，对于较大的编码，SQ 通常一样准确和更快。OPQ 是 vector 的线性变换，使得 vector 更容易压缩。D 是一个维度概念：
- 必须是 M 的倍数
- D <= d，d 是输入向量的维度（这条可选）
- D = 4*M（这条可选）

> 支持 GPU（注意：OPQ转换是在CPU上完成的，但性能并不关键）

## 数据有多大？

这个问题用于填写聚类选项（上面的 `...` 部分）。数据集被聚集到 buckets 中，在搜索时，只访问了一小部分桶（`nprobe`桶这么多）。聚类是在数据集有代表的样本上进行的，我们下面介绍样本的推荐大小。

### 如果低于一百万个 vector：`...,IVF`K`,...`

K是$4*\sqrt(N)$到$16*\sqrt(N)$，N是数据集的大小。其实就是用的 k-means 聚类。您需要 30*K 到256*K 那么多个的向量进行训练（越多越好）。

> 支持 GPU

### 如果1M - 10M：`"...,IVF65536_HNSW32,..."`

IVF 和 HNSW 组合起来，后者做聚类分配。大概需要[30 * 65536,256 * 65536]那么多个 vector 进行训练。

> 不支持 GPU

### 如果10M - 100M：`"...,IVF262144_HNSW32,..."`

与上述相同，将65536替换为262144（2^18）。

请注意，训练会很慢，为了避免这种情况，有两种选择：
- 只做GPU上的训练，其他一切在CPU上运行，请参阅 [train_ivf_with_gpu.ipynb](https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6)。
- 做两层的集群，请参阅 [demo_two_level_clustering.ipynb](https://gist.github.com/mdouze/1b2483d72c0b8984dd152cd81354b7b4)

### 如果100M - 1B：`"...,IVF1048576_HNSW32,..."`

与上述相同，将65536替换为1048576（2^20）。训练会更慢！