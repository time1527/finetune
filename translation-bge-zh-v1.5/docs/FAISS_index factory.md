# FAISS: index factory

**说明：整理自https://github.com/facebookresearch/faiss/wiki/The-index-factory**

index_factory函数将字符串进行解释以生成一个复合的Faiss索引。该字符串是一个逗号分隔的组件列表。它旨在便于构建索引结构，特别是如果它们是嵌套的。index_factory参数通常包括一个预处理组件，一个倒排文件和一个编码组件

`index = index_factory(128, "PCA80,Flat")`：为一个128维向量生成索引，通过PCA降至80维然后进行穷举搜索
`index = index_factory(128, "OPQ16_64,IMI2x8,PQ8+16")`：接受128维向量，将其通过OPQ转换分成16个64维的块，使用2x8位的倒排多索引（= 65536个倒排列表），并使用大小为8的PQ进行细化，然后是16字节

$d$为输入维度

## Prefixes

| String | Class name    | Comments                                                 |
| ------ | ------------- | -------------------------------------------------------- |
| IDMap  | `IndexIDMap`  | 用于在不支持 add_with_ids 的索引上启用它，主要是平面索引 |
| IDMap2 | `IndexIDMap2` | 同上，额外支持 `reconstruct`                             |

## Vector transforms

这些字符串映射到VectorTransform对象，可以在向量被索引之前应用

| String                         | Class name                 | Output dimension | Comments                                                     |
| ------------------------------ | -------------------------- | ---------------- | ------------------------------------------------------------ |
| PCA64, PCAR64, PCAW64, PCAWR64 | `PCAMatrix`                | 64               | 应用PCA变换以减少维度数量。W = follow with whitening, R = follow with random rotation。PCAR在与Flat或标量量化器一起作为预处理时特别有用 |
| OPQ16, OPQ16_64                | `OPQMatrix`                | d, 64            | 将向量旋转，以便它们可以通过PQ有效地编码为16个子向量。输出维度为64，因为通常也有益于减少维度的数量。如果未指定输出维度，则与输入维度相同 |
| RR64                           | `RandomRotation`           | 64               | 输入数据进行随机旋转。维度可能会相对于输入增加或减少         |
| L2norm                         | `NormalizationTransform`   | d                | L2-标准化输入向量                                            |
| ITQ256, ITQ                    | `ITQMatrix`                | 256, d           | 对输入向量应用 ITQ 变换，请参阅 ["Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval" by Gong et al.](http://slazebni.cs.illinois.edu/publications/ITQ.pdf)。当向量使用 LSH 编码时，这非常有用 |
| Pad128                         | `RemapDimensionsTransform` | 128              | 用 0 到 128 维填充输入向量                                   |

## Non-exhaustive search components

### Inverted file indexes

倒排文件都继承自 `IndexIVF`。非穷举组件指定粗粒度量化器（构造函数的第一个参数）应该是什么。index_factory字符串以 IVF 或 IMI 开头，后跟逗号和编码（见下文）

| String           | Quantizer class                | Number of centroids | Comments                                                     |
| ---------------- | ------------------------------ | ------------------- | ------------------------------------------------------------ |
| IVF4096          | `IndexFlatL2` or `IndexFlatIP` | 4096                | 构造具有flat量化器的 IndexIVF 变体之一                       |
| IMI2x9           | `MultiIndexQuantizer`          | 2^(2 * 9) = 262144  | 构建具有更多质心的 IVF，可能更加平衡                         |
| IVF65536_HNSW32  | `IndexHNSWFlat`                | 65536               | 量化器被训练为flat索引，但使用HNSW进行索引。这使得量化速度更快 |
| IVF65536(PQ16x4) | arbitrary                      | 65536               | 使用括号中的字符串构造粗粒度量化器                           |
| IVF1024(RCQ2x5)  | `ResidualCoarseQuantizer`      | 1024                | 这是前一种情况的特例。粗粒度量化器是残差量化器               |

### Graph-based indexes

HNSW 和 NSG 是基于图的索引。它们继承自 `IndexHNSW` 和 `IndexNSG`。两者都依赖于存储实际向量的flat存储 `IndexFlatCodes`

| String            | Storage class          | Comment                                                      |
| ----------------- | ---------------------- | ------------------------------------------------------------ |
| HNSW32, HNSW      | `IndexFlatL2`          | 可以说是最有用的HNSW变体，因为当链接被存储时，压缩向量没有多大意义。 32（每个顶点的链接数）是默认值，可以省略 |
| HNSW32_SQ8        | `IndexScalarQuantizer` | SQ8 标量量化器                                               |
| HNSW32_PQ12       | `IndexPQ`              | PQ12x8索引                                                   |
| HNSW32_16384+PQ12 | `Index2Layer`          | 第一层是flat索引，PQ 对量化器的残差进行编码                  |
| HNSW32_2x10+PQ12  | `Index2Layer`          | 第一层是IMI索引，PQ对量化器的残差进行编码                    |

NSG 变体相同，只是 HNSW 替换为 NSG

### Memory overheads

* 在 `HNSW32` 中，32 编码链接数量，使用最多内存的最低级别有 32 * 2 个链接，即每个向量 32 * 2 * 4 = 256 字节。所以开销是相当可观的
* 在 `NSG32` 中，32 直接编码每个顶点的链接数，因此 32 意味着每个向量 32 * 4 = 128 字节
* 对于 IVF 和 IMI 索引，主要开销是每个向量的 64 位 id 也被存储（即每个向量 8 个字节的开销）

## Encodings

| String                           | Class name (Flat/IVF)                                   | code size (bytes)                       | Comments                                                     |
| -------------------------------- | ------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| Flat                             | `IndexFlat`, `IndexIVFFlat`                             | 4 * d                                   | 这些向量按原样存储，没有进行任何编码                         |
| PQ16, PQ16x12                    | `IndexPQ`, `IndexIVFPQ`                                 | 16, ceil(16 * 12 / 8)                   | 使用 16 个(每个)12位的 Product Quantization 编码。当位数被省略时，默认设置为 8。带有后缀 "np" 的不训练 Polysemous 排列，这可能会很慢 |
| PQ28x4fs, PQ28x4fsr, PQ28x4fs_64 | `IndexPQFastScan`, `IndexIVFPQFastScan`                 | 28 / 2                                  | 与上面的PQ相同，但使用了依赖SIMD指令进行距离计算的PQ的“fast scan”版本。目前仅支持nbits=4。后缀_64表示使用的bbs因子（必须是32的倍数）。后缀fsr（仅适用于IVF）表示向量应通过残差进行编码（速度较慢，更准确 |
| SQ4, SQ8, SQ6, SQfp16            | `IndexScalar` `Quantizer`, `IndexIVF` `ScalarQuantizer` | `4*d/8`, d, `6*d/8`, `d*2`              | 标量量化编码                                                 |
| Residual128, Residual3x10        | `Index2Layer`                                           | `ceil(log2(128) / 8)`, `ceil(3*10 / 8)` | 残差编码。将向量量化为128个质心或3x10个MI质心。Should be followed by PQ or SQ to actually encode the residual. Only for use as a codec. |
| RQ1x16_6x8                       | `IndexResidualQuantizer`                                | (16 + 6*8) / 8                          | 残差量化编解码器。首先使用2^16个质心的量化器对向量进行编码，然后在每个256个质心的6个阶段中对残差进行细化 |
| RQ5x8_Nqint8                     | `IndexResidualQuantizer`                                | 5+1                                     | 残差量化索引，其中范数量化为8位（其他选项是Nqint4和Nfloat）  |
| LSQ5x8_Nqint8                    | `IndexLocalSearchQuantizer`                             | 5+1                                     | 同上，使用局部搜索量化器                                     |
| PRQ16x4x8                        | `IndexProductResidualQuantizer`                         | 16*4                                    | 将向量切分为16个子向量，然后将每个子向量编码为大小为4x8的残差量化器 |
| PLSQ16x4x8                       | `IndexProductResidualQuantizer`                         | 16*4                                    | 同上，使用LSQ量化器                                          |
| ZnLattice3x10_6                  | `IndexLattice`                                          | `ceil((3*log2(C(d/3, 10)) + 6) / 8)`    | Lattice codec. The vector is first split into 3 segments, then each segment is encoded as its norm (6 bits) and as a direction on the unit sphere in dimension d/3, quantized by the Zn lattice of radius^2 = 10. `C(dim, r2)` is the number of points on the sphere of radius sqrt(r2) that have integer coordinates (see [here](https://github.com/facebookresearch/spreadingvectors#zn-quantizer) for more details, and [this Gist](https://gist.github.com/mdouze/b167d9a1d0d8838f3427c68c7d412ad8) on how to set the radius). |
| LSH, LSHrt, LSHr, LSHt           | `IndexLSH`                                              | ceil(d / 8)                             | 通过阈值将向量二值化。在搜索时，查询向量也会被二值化（对称搜索）。后缀 r = 旋转向量先于二值化，t = 训练阈值以平衡 0 和 1。与 ITQ 结合使用效果更佳 |
| ITQ90,SH2.5                      | `IndexIVFSpectralHash`                                  | ceil(90 / 8)                            | 只有 IVF 版本。将向量转换为使用 ITQ（还支持 PCA 和 PCAR）进行编码，将维度减少到 90。然后对每个分量进行编码，[0, 2.5] 表示为位 0，[2.5, 5] 表示为位 1，取模 5。SH2.5g 使用全局阈值，SH2.5c 只使用质心作为阈值。没有参数的 SH 保留向量分量的符号作为位 |

## Suffixes

| String          | Storage class     | Comment                                |
| --------------- | ----------------- | -------------------------------------- |
| RFlat           | `IndexRefineFlat` | 使用精确距离计算重新对搜索结果进行排序 |
| Refine(PQ25x12) | `IndexRefine`     | 同上，但细化索引可以是任何索引。       |

## Example

`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)`：

* `OPQ16_64`: OPQ pre-processing
* `IVF262144(IVF512,PQ32x4fs,RFlat)`: IVF index with 262k centroids. The coarse quantizer is an IVFPQFastScan index with an additional refinement step.
* `PQ16x4fsr`: the vectors are encoded with PQ fast-scan (which takes 16 * 4 / 8 = 8 bytes per vector)
* `Refine(OPQ56_112,PQ56)`: the re-ranking index is a PQ with OPQ pre-processing that occupies 56 bytes.
