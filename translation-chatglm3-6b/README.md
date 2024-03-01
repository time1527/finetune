# README

任务描述：文言文-现代白话文翻译llm微调

llm：ChatGLM3-6B

框架：xtuner

PEFT：QLoRA

GPU：4060Ti 16G

最终目录：（有删减）

```bash
translation-chatglm3-6b
├── SCRIPT.py
├── chat.py
├── chatglm3_6b_qlora.py
├── prepare_data.ipynb
├── raw_data.jsonl
├── train.jsonl
├── test.jsonl
├── ft_res.jsonl
├── unft_res.jsonl
├── hf
│   ├── README.md
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── xtuner_config.py
├── merged
│   ├── config.json
│   ├── configuration_chatglm.py
│   ├── generation_config.json
│   ├── modeling_chatglm.py
│   ├── pytorch_model-00001-of-00007.bin
│   ├── pytorch_model-00002-of-00007.bin
│   ├── pytorch_model-00003-of-00007.bin
│   ├── pytorch_model-00004-of-00007.bin
│   ├── pytorch_model-00005-of-00007.bin
│   ├── pytorch_model-00006-of-00007.bin
│   ├── pytorch_model-00007-of-00007.bin
│   ├── pytorch_model.bin.index.json
│   ├── quantization.py
│   ├── special_tokens_map.json
│   ├── tokenization_chatglm.py
│   ├── tokenizer.model
│   └── tokenizer_config.json
└── work_dirs
    └── chatglm3_6b_qlora
        ├── chatglm3_6b_qlora.py
        ├── epoch_i.pth(表示第i个epoth的pth)
        │   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        │   └── mp_rank_00_model_states.pt
        ├── last_checkpoint
        └── zero_to_fp32.py
```

流程：

1. 环境：

   ```bash
   # 创建虚拟环境
   conda create --name xtuner0.1.9 python=3.10 -y
   conda activate xtuner0.1.9

   # 拉取0.1.9repo
   cd ~
   mkdir xtuner019 && cd xtuner019
   git clone -b v0.1.9  https://github.com/InternLM/xtuner

   # 源码安装
   cd xtuner
   pip install -e '.[all]'
   ```
2. 数据处理与切分：`prepare_data.ipynb`，代码中原始数据用的是绝对路径

   原始数据来自：https://github.com/VMIJUNV/chinese-poetry-and-prose
   
   处理：对于每个样本，提取content和translation
   
   切分：打乱后保留0.95用作训练集，0.05用作和原始模型做效果比较

   ```bash
   cd ~/finetune/translation-chatglm3-6b
   # 运行prepare_data.ipynb
   ```
4. 配置文件：`chatglm3_6b_qlora.py`

   可通过类似 `xtuner copy-cfg chatglm3_6b_qlora_oasst1_e3 .`将内置配置复制到当前路径再做修改

   主要修改：

   1. 模型路径：`pretrained_model_name_or_path = "/home/dola/model/ZhipuAI/chatglm3-6b"`
   2. 数据路径：`data_path = "/home/dola/finetune/translation-chatglm3-6b/train.jsonl"`
   3. 优化器(wsl2)：`optim_type = AdamW # PagedAdamW32bit`
   4. system：
      ```python
      SYSTEM = '你是一名文言文翻译家，能够将文言文准确优雅地翻译成现代白话文。'
      evaluation_inputs = [
          '请将下述文言文翻译成现代白话文：忽如一夜春风来，千树万树梨花开', 
          '请将下述文言文翻译成现代白话文：但愿人长久，千里共婵娟',
          "请将下述文言文翻译成现代白话文：儿童散学归来早，忙趁东风放纸鸢"
      ]
      ```
   5. train_dataset：
      1. `dataset=dict(type=load_dataset, path="json",data_files=dict(train=data_path))`
      2. `dataset_map_fn=None`
5. 模型：本地路径 `/home/dola/model/ZhipuAI/chatglm3-6b`
6. 微调：

   ```bash
   xtuner train ./chatglm3_6b_qlora.py --deepspeed deepspeed_zero2
   ```
7. 模型转换与合并：

   ```bash
   # 转换:xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
   mkdir hf
   export MKL_SERVICE_FORCE_INTEL=1
   xtuner convert pth_to_hf ./chatglm3_6b_qlora.py ./work_dirs/chatglm3_6b_qlora/epoch_10.pth ./hf

   # 合并
   # SCRIPT.py来自:https://github.com/InternLM/xtuner/issues/273
   python SCRIPT.py /home/dola/model/ZhipuAI/chatglm3-6b ./hf ./merged --max-shard-size 2GB

   # xtuner convert merge \
   #     ${NAME_OR_PATH_TO_LLM} \
   #     ${NAME_OR_PATH_TO_ADAPTER} \
   #     ${SAVE_PATH} \
   #     --max-shard-size 2GB
   ```
8. 对话：`chat.py`改自 `xtuner chat`，注意固定了测试集路径

   ```bash
   python chat.py ./merged --prompt-template chatglm3 --system "你是一名文言文翻译家，能够将文言文准确优雅地翻译成现代白话文。" --save_path ./ft_res.jsonl
   ```

参考：

1. https://github.com/VMIJUNV/chinese-poetry-and-prose
2. https://github.com/InternLM/xtuner/tree/v0.1.9
3. https://github.com/InternLM/tutorial
