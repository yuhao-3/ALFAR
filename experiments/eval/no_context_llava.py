"""
No-context inference for InfoSeek/ViQuAE - Motivation Experiment
This script runs the same model WITHOUT retrieved context to establish baseline performance.
"""
import argparse
import json
from tqdm import tqdm
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import set_seed
from vcd_sample import evolve_vcd_sampling
import numpy as np
from pathlib import Path

# Use same sampling mechanism as ALFAR (but without contrastive decoding)
evolve_vcd_sampling()

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / 'data'

    if args.dataset == 'infoseek':
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'infoseek_mc.json'
    elif args.dataset == 'viquae':
        question_file = DATA_DIR / 'eval_data' / 'mc' / 'viquae_mc.json'
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    with open(os.path.expanduser(question_file), "r") as f:
        questions = json.load(f)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line['id']
        question = line["question"]
        choices = line['multiple_choices']

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Build prompt WITHOUT context - parametric knowledge only
        conv = conv_templates[args.conv_mode].copy()
        temp = '\n Option: \n'
        for c_name, c_content in choices.items():
            temp += f"{c_name}: {c_content}\n"

        # Key difference: NO context, just question and options
        conv.append_message(conv.roles[0], qs + ' Answer the question using a single word based on your knowledge.' + temp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if args.dataset == 'infoseek':
            image_file = line["image"].split('.')[0]
            try:
                raw_image = Image.open(os.path.join(args.image_folder, image_file + '.jpg')).convert('RGB')
            except:
                raw_image = Image.open(os.path.join(args.image_folder, image_file + '.JPEG')).convert('RGB')
        else:
            image_file = line["image"]
            raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        # Simple generation without ALFAR modifications
        # Use same sampling as ALFAR (evolve_vcd_sampling) but without contrastive decoding
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                att_alpha=0.0,  # Disable ALFAR attention reallocation
                img_start_idx=35,
                img_end_idx=611,
                question_len=0,
                prompt_len=0,
                context_len=0,
                ret_sim=0.0,
                do_sample=True,  # Use sampling (same as ALFAR)
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=1,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        outputs = outputs.strip()

        ans_file.write(json.dumps({"data_id": idx,
                                   "prediction": outputs,
                                   "model_id": model_name,
                                   "image": image_file}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/gpfs/projects/punim2075/model_hub/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/gpfs/projects/punim2075/ALFAR/data/images")
    parser.add_argument("--dataset", type=str, default="infoseek", choices=["infoseek", "viquae"])
    parser.add_argument("--answers-file", type=str, default="../result/no_context_infoseek_results.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
