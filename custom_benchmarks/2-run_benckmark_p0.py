# Prompt Engineering Experiments — Step 2 (P0 Runner)
# SPICE Netlist Generation (ngspice-compatible)

## This script runs the P0 benchmark generation ##
#    loops through benchmark 1
#    builds the P0 prompt with build_prompt(entry)
#    generates
#    extracts the completion-only netlist
#    writes runs/P0/<case_id>.cir

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

GEN_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 200,  # keep identical across P0/P1/P2/P3 for fairness
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 6,
}

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(entry):
    # baseline instruction prompt (P0)
    return (
        "Write an ngspice-compatible SPICE netlist.\n"
        "Output ONLY the netlist text (no explanation, no markdown, no code fences).\n"
        "Use node 0 as ground.\n"
        "The final line must be exactly: .end\n\n"
        f"Request: {entry['spec']}\n"
    )

def text_to_token_ids(text, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs.get("attention_mask", None)

def extract_netlist(text):
    t = text.strip()

    # prefer fenced code if it exists
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()

    # truncate after ".end" if present
    lower = t.lower()
    end_idx = lower.find(".end")
    if end_idx != -1:
        t = t[: end_idx + len(".end")].strip()

    return t

def generate(model, input_ids, attention_mask, gen_config, tokenizer):
    # generation wrapper
    cfg = dict(gen_config)

    # make generate() happy across models and tokenizers
    cfg["pad_token_id"] = tokenizer.pad_token_id
    cfg["eos_token_id"] = tokenizer.eos_token_id

    if attention_mask is None:
        out = model.generate(input_ids=input_ids, **cfg)
    else:
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **cfg)

    return out

def decode_completion_only(out_ids, prompt_input_ids, tokenizer):
    # decode only the newly generated completion - exclude prompt tokens
    gen_only = out_ids[0][prompt_input_ids.shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


## main runner loop (Step 2)
def main():
    torch.manual_seed(123) # 123 for reproducibility

    BENCHMARK_PATH = Path("./benchmark/spice_benchmark.json")
    out_dir = Path("./runs/P0")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(benchmark_path)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of test case objects.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    # avoid generate() warnings if pad token is not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model.device

    for i, entry in enumerate(data, start=1):
        case_id = entry.get("id", f"case_{i:03d}")

        input_text = build_prompt(entry)
        input_ids, attention_mask = text_to_token_ids(input_text, tokenizer, device)

        try:
            out_ids = generate(model, input_ids, attention_mask, GEN_CONFIG, tokenizer)
            completion = decode_completion_only(out_ids, input_ids, tokenizer)
            netlist_text = extract_netlist(completion)
        except Exception as e:
            netlist_text = f" - Error generating netlist for {case_id}: {e}\n.end"

        out_path = out_dir / f"{case_id}.cir" # <-- SPICE circuit file extension
        out_path.write_text(netlist_text + "\n", encoding="utf-8")

        print(f"[{i}/{len(data)}] wrote {out_path}")

    print("Complete")

if __name__ == "__main__":
    main()
