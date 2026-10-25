"""QLoRA SFT on expert-iteration data, loss on the assistant response only.

Uses TRL's conversational prompt-completion format so loss is computed only
on the completion (the previous run trained on full chat text, so most of
the loss went to predicting the prompt). --init-adapter continues training
an existing adapter (round r builds on round r-1).
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for p in args.data:
        for l in open(p):
            ex = json.loads(l)
            m = ex["messages"]
            rows.append({"prompt": m[:2], "completion": m[2:]})
    if args.max_examples and len(rows) > args.max_examples:
        import random
        random.Random(0).shuffle(rows)
        rows = rows[:args.max_examples]
    print(f"{len(rows)} training examples")
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(args.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="auto", dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    peft_config = None
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=args.lr, logging_steps=10,
        save_strategy="no", bf16=True, max_length=1536, report_to=[],
        gradient_checkpointing=True, completion_only_loss=True, lr_scheduler_type="cosine",
        warmup_steps=5,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
