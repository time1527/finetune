# README

任务描述：现代白话文-诗词曲embedding微调

embedding：bge-zh-v1.5

GPU：4060Ti 16G

最终目录：（有删减和顺序改动）

```bash
translation-bge-zh-v1.5
├── prepare_data.ipynb
├── raw_data.jsonl
├── candidate_pool.jsonl
├── total_minedHN.jsonl
├── split_data.ipynb
├── train.jsonl
├── eval.jsonl
├── ft-bge-zh-v1.5
│   ├── 1_Pooling
│   │   └── config.json
│   ├── 2_Normalize
│   ├── README.md
│   ├── added_tokens.json
│   ├── checkpoint-i
│   │   ├── 1_Pooling
│   │   │   └── config.json
│   │   ├── 2_Normalize
│   │   ├── README.md
│   │   ├── added_tokens.json
│   │   ├── config.json
│   │   ├── config_sentence_transformers.json
│   │   ├── modules.json
│   │   ├── optimizer.pt
│   │   ├── pytorch_model.bin
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── sentence_bert_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   └── vocab.txt
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── modules.json
│   ├── pytorch_model.bin
│   ├── runs
│   │   └── Mar01_14-59-59_DESKTOP-99S7A0E
│   │       └── events.out.tfevents.1709276402.DESKTOP-99S7A0E.92147.0
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   └── vocab.txt
└──  tmp.sh
```

流程：

1. 环境：

   ```bash
   # 在原来的虚拟环境在进行的
   conda activate xtuner0.1.9

   # 拉取FlagEmbedding repo
   cd ~
   git clone https://github.com/FlagOpen/FlagEmbedding.git

   # 源码安装
   cd FlagEmbedding
   pip install -e .
   ```
2. 数据：`prepare_data.ipynb`，使用的是预处理过的数据，绝对路径

   原始数据来自：https://github.com/VMIJUNV/chinese-poetry-and-prose

   得到数据 `raw_data.jsonl`和 `candidate_pool.jsonl`

   ```bash
   cd ~/finetune/translation-bge-zh-v1.5
   # 运行prepare_data.ipynb
   ```
3. 模型：本地路径 `/home/dola/embedding/BAAI/bge-base-zh-v1.5`
4. 难负例挖掘：

   ```bash
   python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
   --model_name_or_path /home/dola/embedding/BAAI/bge-base-zh-v1.5 \
   --input_file /home/dola/finetune/translation-bge-zh-v1.5/raw_data.jsonl \
   --output_file /home/dola/finetune/translation-bge-zh-v1.5/total_minedHN.jsonl \
   --candidate_pool /home/dola/finetune/translation-bge-zh-v1.5/candidate_pool.jsonl \
   --range_for_sampling 2-20 \
   --negative_number 9 
   ```
5. 修改源码：加上验证部分

   说明：没有使用 `load_best_model_at_end`（报错短时间内未解决）及会受其影响的 `EarlyStoppingCallback`
6. 微调：

   ```bash
   torchrun --nproc_per_node 1 \
   -m FlagEmbedding.baai_general_embedding.finetune.run \
   --output_dir /home/dola/finetune/translation-bge-zh-v1.5/ft-bge-zh-v1.5 \
   --model_name_or_path /home/dola/embedding/BAAI/bge-base-zh-v1.5 \
   --train_data /home/dola/finetune/translation-bge-zh-v1.5/train.jsonl \
   --learning_rate 2e-5 \
   --fp16 \
   --num_train_epochs 50 \
   --per_device_train_batch_size 64 \
   --dataloader_drop_last True \
   --normlized True \
   --temperature 0.02 \
   --query_max_len 64 \
   --passage_max_len 32 \
   --train_group_size 10 \
   --eval_group_size 10 \
   --negatives_cross_device \
   --logging_steps 10 \
   --query_instruction_for_retrieval "" \
   --eval_steps 100 \
   --evaluation_strategy steps \
   --metric_for_best_model eval_loss \
   --eval_data /home/dola/finetune/translation-bge-zh-v1.5/eval.jsonl \
   --save_steps 500
   ```
7. 损失：为使训练损失和验证损失较好地在同一张图展示，对验证损失做了缩放（不影响相对大小）
   
   ![image](https://github.com/time1527/finetune/assets/154412155/d5bdd574-41b8-4d0e-aaf6-176becc6b2d0)

参考：

1. https://github.com/VMIJUNV/chinese-poetry-and-prose
2. https://github.com/FlagOpen/FlagEmbedding
