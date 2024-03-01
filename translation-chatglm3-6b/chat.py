# Copyright (c) OpenMMLab. All rights reserved.
# copy from xtuner.tools.chat.py and modified
import argparse
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)

from xtuner.tools.utils import get_chat_utils, update_stop_criteria
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
import json
import jsonlines

os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    
    parser.add_argument(
        'model_name_or_path', 
        help='Hugging Face model name or path')
    
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', 
        default=None, 
        help='Specify the system text')

    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    
    parser.add_argument(
        '--save_path',
        help='the jsonl path to save the model inference result')
    
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument('--command-stop-word', default=None, help='Stop key')
    parser.add_argument('--answer-stop-word', default=None, help='Stop key')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help='repetition_penalty')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # model_kwargs
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'trust_remote_code': True
    }


    # build model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                    **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)

    model.eval()

    Streamer, stop_criteria = get_chat_utils(model)
    if args.no_streamer:
        Streamer = None

    command_stop_cr, answer_stop_cr = update_stop_criteria(
        base=stop_criteria,
        tokenizer=tokenizer,
        command_stop_word=args.command_stop_word,
        answer_stop_word=args.answer_stop_word)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        repetition_penalty = args.repetition_penalty
    )

    print(f"will save in {args.save_path}")
    ft_res = []
    with open("test.jsonl") as file:
        for item in jsonlines.Reader(file):
            text = item["conversation"][0]["input"]
            save_ref = item["conversation"][0]["output"]
            save_input = "".join(text.split("：")[1:])
            inputs = ''
            if args.prompt_template:
                prompt_text = ''
                template = PROMPT_TEMPLATE[args.prompt_template]
                if 'SYSTEM' in template:
                    system_text = None
                    if args.system is not None:
                        system_text = args.system
                    if system_text is not None:
                        prompt_text += template['SYSTEM'].format(
                            system=system_text)
                prompt_text += template['INSTRUCTION'].format(input=text)
            inputs += prompt_text
            print(inputs)

            # chatglm3-6b最长8k
            if len(inputs) >= 8000:            
                continue
            save_total_inputs = inputs
            ids = tokenizer.encode(inputs, return_tensors='pt')    

            generate_output = model.generate(
                inputs=ids.cuda(),
                generation_config=gen_config,
                stopping_criteria=answer_stop_cr,
                )
            output_text = tokenizer.decode(generate_output[0][len(ids[0]):])
            print(output_text)
            save_output = output_text.strip()
            ft_res.append({"input":save_input,"output":save_output,"ref":save_ref,"total_inputs":save_total_inputs})
            print("====" * 10)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        for item in ft_res:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str+"\n")

if __name__ == '__main__':
    main()
