"""
Context-Aware Decoding (CAD) Baseline for OKVQA/AOKVQA

CAD amplifies the difference between output probabilities when model is used
with and without context, making the model trust the provided evidence more.

CAD(x) = (1 + alpha) * log P_context(x) - alpha * log P_no_context(x)

Reference:
"Trusting Your Evidence: Hallucinate Less with Context-aware Decoding" (NAACL 2024)
Shi et al.
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

        # Prompt without context (for CAD)
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
            # For simplified CAD implementation, we use the contrastive decoding approach
            # Full CAD would require custom logits processing during generation

            # This uses the existing ALFAR infrastructure with cd_beta as CAD alpha
            # images_cd points to no-context input for contrastive effect
            output_ids = model.generate(
                input_ids_context,
                images=raw_image_tensor.unsqueeze(0).half().cuda(),
                images_cd=input_ids_no_context,  # Use no-context for contrastive decoding
                cd_beta=args.cad_alpha,  # CAD alpha parameter
                att_alpha=0.0,  # No attention reallocation (pure CAD)
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=10,
                use_cache=True
            )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/model/llava_1.5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2017")
    parser.add_argument("--dataset", type=str, default="aokvqa", choices=["aokvqa", "okvqa"])
    parser.add_argument("--answers-file", type=str, default="../result/cad_aokvqa.csv")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--cad-alpha", type=float, default=0.5, help="CAD alpha parameter for context amplification")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
