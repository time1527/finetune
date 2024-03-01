# BGE

**说明：整理自https://github.com/FlagOpen/FlagEmbedding**

## Embedding

### 一些问题

1. 效果差：与其他使用mean pooling的嵌入模型不同，BGE使用[cls]的最后一个隐藏状态作为句子嵌入：`sentence_embeddings = model_output[0][:, 0]`。如果使用mean pooling，性能会显著下降
2. 如何微调：根据[这个例子](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)
   * 根据[这个例子](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives)挖掘更多的难负例，可以提高检索性能
   * 一般来说，增大`per_device_train_batch_size`会带来更好的性能。可以通过启用 `--fp16`，`--deepspeed df_config.json`，`--gradient_checkpointing` 等来扩展
   * 如果想在微调时保持在其他任务上的性能，可以使用 LM-Cocktail 将微调模型和原始 bge 模型合并。此外，如果想在多个任务上进行微调，也可以通过模型合并来近似多任务学习，如 LM-Cocktail
   * 如果在数据上预训练 bge，那么预训练模型不能直接用于计算相似度，它必须在计算相似度之前用对比学习进行微调
   * 如果微调模型的准确度仍然不高，建议使用/微调交叉编码器模型（bge-reranker）来重新排名前$k$个结果，同时也需要难负例来微调reranker
3. 两个不相似句子之间的相似度分数高于0.5：建议使用 bge v1.5，这可以缓解相似度分布的问题。由于使用temperature为0.01的对比学习来微调模型，当前BGE模型的相似度分布大约在[0.6, 1]的区间内。因此，相似度分数大于0.5并不表示这两个句子相似。对于下游任务，如段落检索或语义相似度，重要的是分数的相对顺序，而不是绝对值。如果需要基于相似度阈值筛选相似句子，需要根据数据上的相似度分布（如0.8、0.85甚至0.9）选择适当的相似度阈值
4. query instruction需要在什么时候使用：对于 bge-*-v1.5，当不使用指令时，我们提高了它的检索能力。与使用指令相比，没有指令只会导致检索性能略微下降。因此，为了方便起见，可以在所有情况下生成不带指令的嵌入。对于使用短查询来查找长相关文档的检索任务，建议为这些短查询添加指令。决定是否为查询添加指令的最佳方法是选择在任务上实现更好性能的设置。在所有情况下，文档/段落不需要添加指令

### 使用

#### FlagEmbedding

```bash
pip install -U FlagEmbedding
```

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```

#### Sentence-Transformers

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

```python
# 对于s2p（短查询到长段落）检索任务，每个短查询都应以一条指示开始（指示请参见模型列表）。但是段落不需要指示。
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

#### Langchain

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```

#### Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```

### 微调数据格式

`{"query": str, "pos": List[str], "neg":List[str]}`

query是查询，pos 是一系列正文本，neg 是一系列负文本。如果一个查询没有负文本，可以从整个语料库中随机抽取一些作为负文本

### Hard Negatives

```shell
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

* `input_file`: 用于微调的json数据。该脚本将为每个查询检索前$k$个文档，并从前$k$个文档中随机抽样负文档（不包括正文档）
* `output_file`: 挖掘硬负例后，输出的 JSON 数据路径
* `negative_number`: 采样的负样本数
* `range_for_sampling`：在哪里采样负例。例如，`2-100` 表示从前2到前200个文档中采样`negative_number`个负例，可以设置更大的值来减少负例的难度（例如，将其设置为60-300，从前60到前300个段落中采样负例）
* `candidate_pool`: 检索池。默认值为None，这时此脚本将从`input_file`中的所有`neg`的组合中检索。该文件的格式与[预训练数据](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format)相同。如果输入了候选池，此脚本将从该文件中检索负例

* `use_gpu_for_searching`: 是否使用faiss-gpu来检索负样本

### FineTune

```shell
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 
```

- `per_device_train_batch_size`: 在训练中的批量大小。在大多数情况下，更大的batch size会带来更强的性能。可以通过启用 `--fp16`，`--deepspeed ./df_config.json`（df_config.json 可以参考 [ds_config.json](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/ds_config.json)），`--gradient_checkpointing` 等来增加batch size

- `train_group_size`: 在训练中，查询的正负样本数量。始终有一个正样本，因此该参数将控制负样本的数量（#负样本=`train_group_size`-1）。请注意，负样本的数量不应大于数据 `"neg":List[str]`中的负样本数量。除了该组中的负样本外，批内的负样本也将用于微调

- `negatives_cross_device`: 在所有的GPU上共享负样本，这个参数将扩大负样本的数量

- `learning_rate`: 选择适合模型的学习率。建议大型/基础/小型规模分别使用1e-5/2e-5/3e-5

- `temperature`: 影响相似度分数的分布
  
  - https://github.com/FlagOpen/FlagEmbedding/issues/155
  
  - ```python
    # FlagEmbedding/FlagEmbedding/baai_general_embedding/finetune/modeling.py
    # BiEncoderModel.forward
    if self.training:
        if self.negatives_cross_device and self.use_inbatch_neg:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
    
        group_size = p_reps.size(0) // q_reps.size(0)
        if self.use_inbatch_neg:
            scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
            scores = scores.view(q_reps.size(0), -1) # (B,B*group_size)
    
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * group_size
            loss = self.compute_loss(scores, target)
        else:
            scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G
    
            scores = scores.view(q_reps.size(0), -1)
            target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            loss = self.compute_loss(scores, target)
    
    else:
        scores = self.compute_similarity(q_reps, p_reps) #(B,B*group_size) 
        loss = None
    ```
  
  - 计算loss的时候score会除以temperature，所以temperature为0.01的时候，相当于把分数放大了100倍。score微小的变动就会造成loss的变化，模型更好优化，但同时推理时不同样本的相似度score差别会比较小，因为训练的时候把这个差别放大了100倍
  
  - temperature越大，模型效果会越差，建议在0.01-0.05之间
  
- `query_max_len`: 查询的最大长度，根据数据中查询的平均长度进行设置

- `passage_max_len`: 段落的最大长度，根据数据中段落的平均长度进行设置

- `query_instruction_for_retrieval`: query的instruction，会被加在每个query之前。`""`表示什么都不加

- `use_inbatch_neg`: 使用同一批次的段落作为负样本。默认值为True

[TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)其他：【修改验证代码时可能用得到】

* `evaluation_strategy`：训练过程中采用的评估策略。可能的取值包括：
  * "no"：训练过程中不进行评估
  * "steps"：每 `eval_steps` 次训练进行评估（并记录结果）
  * "epoch"：每个周期结束时进行评估
* `eval_steps`：两次评估之间的更新步数，如果`evaluation_strategy="steps"`。如果未设置，将默认为与logging_steps相同的值。应该是一个整数或范围为[0,1)的浮点数。如果小于1，将被解释为总训练步数的比率

* `do_eval`：是否在验证集上运行评估。如果evaluation_strategy与"no"不同，则设置为True。Trainer不直接使用此参数，而是打算由您的训练/评估脚本使用
* `load_best_model_at_end`：是否在训练结束时加载找到的最佳模型。启用此选项时，最佳检查点将始终被保存
  * 当设置为True时，参数`save_strategy`需要与`evaluation_strategy`相同，且在`save_strategy`为“steps”时，`save_steps`必须是`eval_steps`的整数倍
  * `transformers.EarlyStoppingCallback`：这个回调函数依赖于TrainingArguments参数`load_best_model_at_end`的功能来在TrainerState中设置`best_metric`
* `metric_for_best_model`：与`load_best_model_at_end`一起使用，指定用于比较两个不同模型的指标。必须是评估返回的指标的名称，可以带有前缀"eval_"，也可以不带。如果未指定，并且`load_best_model_at_end=True`（使用评估损失），则默认为"loss"
* `save_strategy`：训练期间采用的检查点保存策略。可能的取值包括：默认是steps
  * "no"：训练期间不保存检查点
  * "epoch"：在每个周期结束时保存
  * "steps"：每`save_steps`次保存一次

* `gradient_accumulation_steps `：更新梯度之前累积梯度的步数
  * 在使用梯度累积时，一个步骤被计为一个带有反向传播的步骤。因此，日志记录、评估和保存将在每个`gradient_accumulation_steps * xxx_step`训练样本后进行

## issue综合

1. embedding和reranker微调效果：https://github.com/FlagOpen/FlagEmbedding/issues/296

   首先，基于bge-large-zh-1.5，难负例微调，微调一个embedding模型；再基于这个微调后的embedding模型，难负例挖掘，微调reranker模型
   另外，第二步的neg是80，第一步当时没做多参数的对比测试，用的默认数量，15
   
2. embedding微调评估：https://github.com/ninehills/blog/issues/104