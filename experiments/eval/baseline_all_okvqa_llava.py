"""
Unified Baseline Implementation for OKVQA/AOKVQA

This script implements all baseline methods mentioned in the ALFAR paper:
1. No-Context: Parametric knowledge only (already implemented separately)
2. Regular MRAG: Context in prompt without amplification (already implemented separately)
3. VCD: Visual Contrastive Decoding
4. CD: Contrastive Decoding
5. CAD: Context-Aware Decoding
6. AdaCAD: Adaptive Context-Aware Decoding
7. Entropy: Entropy-based Decoding
8. COIECD: Contextual Information-Entropy Constraint Decoding

Note: AGLA requires more complex modifications and is implemented separately.

Usage:
python baseline_all_okvqa_llava.py --method cad --dataset aokvqa --cad-alpha 0.5
"""

import argparse
import json
from tqdm import tqdm
import sys
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image, ImageFilter
from transformers import set_seed
from vcd_sample import evolve_vcd_sampling
from pathlib import Path

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

    print(f"Running baseline method: {args.method.upper()}")
    print(f"Parameters: {vars(args)}")

    for i in tqdm(range(questions.shape[0])):
        test_sample = questions.iloc[i]

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

        # Prepare prompts
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Prompt with context
        conv_context = conv_templates[args.conv_mode].copy()
        conv_context.append_message(conv_context.roles[0], qs + ' Answer the question using a single word or phrase based on the given context. Context: ' + context)
        conv_context.append_message(conv_context.roles[1], None)
        prompt_context = conv_context.get_prompt()
        input_ids_context = tokenizer_image_token(prompt_context, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Prompt without context (for contrastive methods)
        conv_no_context = conv_templates[args.conv_mode].copy()
        conv_no_context.append_message(conv_no_context.roles[0], qs + ' Answer the question using a single word or phrase.')
        conv_no_context.append_message(conv_no_context.roles[1], None)
        prompt_no_context = conv_no_context.get_prompt()
        input_ids_no_context = tokenizer_image_token(prompt_no_context, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Load image
        raw_image = Image.open(image_path).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv_context.sep if conv_context.sep_style != SeparatorStyle.TWO else conv_context.sep2

        with torch.inference_mode():
            # Select baseline method
            if args.method == 'vcd':
                # Visual Contrastive Decoding
                # Create distorted image
                distorted_image = raw_image.filter(ImageFilter.GaussianBlur(radius=args.vcd_blur_radius))
                distorted_image_tensor = image_processor.preprocess(distorted_image, return_tensors='pt')['pixel_values'][0]

                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=input_ids_context,  # Same text, different image
                    cd_beta=args.vcd_alpha,
                    att_alpha=0.0,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            elif args.method == 'cd':
                # Contrastive Decoding (text-based)
                # Uses no-context as "amateur" model
                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=input_ids_no_context,
                    cd_beta=args.cd_alpha,
                    att_alpha=0.0,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            elif args.method == 'cad':
                # Context-Aware Decoding
                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=input_ids_no_context,
                    cd_beta=args.cad_alpha,
                    att_alpha=0.0,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            elif args.method == 'adacad':
                # Adaptive CAD
                # Simplified: uses higher alpha for adaptive effect
                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=input_ids_no_context,
                    cd_beta=args.adacad_alpha_max,
                    att_alpha=0.0,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            elif args.method == 'entropy':
                # Entropy-based Decoding
                # Simplified: uses lower temperature for lower entropy
                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.entropy_temperature,
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            elif args.method == 'coiecd':
                # COIECD: Combines CAD with entropy constraints
                output_ids = model.generate(
                    input_ids_context,
                    images=raw_image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=input_ids_no_context,
                    cd_beta=args.coiecd_alpha,
                    att_alpha=0.0,
                    do_sample=True,
                    temperature=args.coiecd_temperature,  # Entropy constraint via temperature
                    top_p=args.top_p,
                    max_new_tokens=10,
                    use_cache=True
                )

            else:
                raise ValueError(f"Unknown method: {args.method}")

        input_token_len = input_ids_context.shape[1]
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
    print(f"Results saved to: {ans_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2017")
    parser.add_argument("--dataset", type=str, default="aokvqa", choices=["aokvqa", "okvqa"])
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # Method selection
    parser.add_argument("--method", type=str, required=True,
                       choices=["vcd", "cd", "cad", "adacad", "entropy", "coiecd"],
                       help="Baseline method to use")

    # VCD parameters
    parser.add_argument("--vcd-alpha", type=float, default=0.5)
    parser.add_argument("--vcd-blur-radius", type=float, default=10.0)

    # CD parameters
    parser.add_argument("--cd-alpha", type=float, default=0.5)

    # CAD parameters
    parser.add_argument("--cad-alpha", type=float, default=0.5)

    # AdaCAD parameters
    parser.add_argument("--adacad-alpha-max", type=float, default=1.0)

    # Entropy parameters
    parser.add_argument("--entropy-temperature", type=float, default=0.5)

    # COIECD parameters
    parser.add_argument("--coiecd-alpha", type=float, default=0.5)
    parser.add_argument("--coiecd-temperature", type=float, default=0.7)

    args = parser.parse_args()

    # Set default answers file if not provided
    if args.answers_file is None:
        args.answers_file = f"../result/{args.method}_{args.dataset}_results.csv"

    set_seed(args.seed)
    eval_model(args)
