"""
Regular MRAG (Standard RAG) for OKVQA/AOKVQA - Motivation Experiment Control

This is the CRITICAL missing baseline!

Three-way comparison:
1. No-Context:    No context + No ALFAR
2. Regular MRAG:  Has context + No ALFAR (THIS SCRIPT)
3. ALFAR:         Has context + Has ALFAR

Regular MRAG = Standard RAG approach:
- Context is concatenated to the prompt
- Model generates normally (no contrastive decoding)
- No attention reallocation
- This isolates the effect of ALFAR's amplification vs just having context
"""

import argparse
import json
from tqdm import tqdm
import sys
import os
import pandas as pd
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
from pathlib import Path

# Use same sampling as ALFAR (for fair comparison)
evolve_vcd_sampling()


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / 'data' / 'eval_data' / 'okvqa'

    if args.dataset == 'aokvqa':
        questions = pd.read_csv(DATA_DIR / 'a_ok_vqa_val_fixed_annots.csv')
        knowledge = json.load(open(DATA_DIR / 'aokvqa_val_dcaption.json'))
    elif args.dataset == 'okvqa':
        questions = pd.read_csv(DATA_DIR / 'val_annots_fixed.csv')
        knowledge = json.load(open(DATA_DIR / 'okvqa_val_dcaption.json'))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    question_id_list = []
    image_id_list = []
    ans_list = []

    for i in tqdm(range(questions.shape[0])):
        test_sample = questions.iloc[i]

        # Load retrieved context (same as ALFAR)
        context = knowledge[str(test_sample.question_id)]
        idx = test_sample.question_id
        image_file = test_sample.image_path

        # Handle different image filename formats
        if image_file.startswith('COCO_val2014_'):
            image_file = image_file.replace('COCO_val2014_', '')
        question = test_sample.question

        # Check if image exists
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping question {idx}")
            continue

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Build prompt WITH context (same as ALFAR)
        # But we will NOT use contrastive decoding or attention reallocation
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0],
                          qs + 'Answer the question using a single word or phrase based on the given context. Context: ' + context)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        raw_image = Image.open(image_path).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # Standard generation WITHOUT ALFAR interventions
        # KEY DIFFERENCES from ALFAR:
        # 1. NO images_cd (no contrastive decoding)
        # 2. att_alpha = 0 (no attention reallocation)
        # 3. Context is in prompt but processed normally
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                # NO images_cd parameter → no contrastive decoding
                att_alpha=0.0,  # Disable ALFAR attention reallocation
                img_start_idx=35,
                img_end_idx=611,
                question_len=0,  # Not used without ALFAR
                prompt_len=0,    # Not used without ALFAR
                context_len=0,   # Not used without ALFAR
                ret_sim=0.0,     # Not used without ALFAR
                do_sample=True,  # Same sampling as ALFAR
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=10,
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

        question_id_list.append(test_sample.question_id)
        image_id_list.append(test_sample.image_id)
        ans_list.append(outputs)

    llama_preds_df = pd.DataFrame({'question_id': question_id_list, 'image_id': image_id_list, 'llama_answer': ans_list})
    llama_preds_df['llama_answer'] = llama_preds_df['llama_answer'].apply(lambda x: x.replace("=", "").strip())
    llama_preds_df.to_csv(ans_file, index=False)
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/gpfs/projects/punim2075/model_hub/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/gpfs/projects/punim2075/ALFAR/data/images/coco/val2014")
    parser.add_argument("--dataset", type=str, default="aokvqa", choices=["aokvqa", "okvqa"])
    parser.add_argument("--answers-file", type=str, default="../result/regular_mrag_aokvqa_results.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
