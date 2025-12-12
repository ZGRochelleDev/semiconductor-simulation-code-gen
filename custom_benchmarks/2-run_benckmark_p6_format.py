# Prompt Engineering Experiments — Step 2 (P2 Runner)
# P2: Few-shot prompting (3 curated examples)

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

def load_and_parse_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))

def few_shot_block():
    # 3 examples (OP / TRAN / AC), im trying to keep these short and constant
    return (
        "You are an expert ngspice netlist generator.\n"
        "Return ONLY a valid ngspice-compatible SPICE netlist as plain text.\n"
        "No explanations, no markdown, no code fences.\n"
        "Use node 0 as ground. The final line must be exactly: .end\n"
        "Always include at least one output line: .print or .plot or .meas\n\n"

        "### Example 1 (OP)\n"
        "Request: Create a DC divider. V1 is 5V DC from in to 0. R1 is 1k from in to out. "
        "R2 is 1k from out to 0. Run .op and print V(out).\n"
        "Netlist:\n"
        "* divider_op\n"
        "V1 in 0 DC 5\n"
        "R1 in out 1k\n"
        "R2 out 0 1k\n"
        ".op\n"
        ".print V(out)\n"
        ".end\n\n"

        "### Example 2 (TRAN)\n"
        "Request: RC step response. VIN is PULSE(0 3.3 0 1n 1n 50n 100n) from in to 0. "
        "R1 is 2k from in to out. C1 is 1n from out to 0. Run .tran 1n 200n and print V(out).\n"
        "Netlist:\n"
        "* rc_tran\n"
        "VIN in 0 PULSE(0 3.3 0 1n 1n 50n 100n)\n"
        "R1 in out 2k\n"
        "C1 out 0 1n\n"
        ".tran 1n 200n\n"
        ".print tran V(out)\n"
        ".end\n\n"

        "### Example 3 (AC)\n"
        "Request: RC low-pass filter. V1 is AC 1V from in to 0. R1 is 10k from in to out. "
        "C1 is 10n from out to 0. Run .ac dec 50 10 100k and print V(out).\n"
        "Netlist:\n"
        "* rc_ac\n"
        "V1 in 0 AC 1\n"
        "R1 in out 10k\n"
        "C1 out 0 10n\n"
        ".ac dec 50 10 100k\n"
        ".print ac V(out)\n"
        ".end\n\n"
    )

def build_prompt_format_control(entry):
    return (
        few_shot_block()
        +
        "### Output Format Rules\n"
        "Wrap the final ngspice netlist between the tags [NETLIST] and [/NETLIST].\n"
        "Inside [NETLIST]...[/NETLIST], write ONLY valid ngspice lines:\n"
        "- No explanations, comments, or markdown.\n"
        "- Use node 0 as ground when appropriate.\n"
        "- Always include any required analysis and at least one .print/.plot/.meas.\n"
        "- Always end the netlist with a single .end line.\n\n"
        "Anything outside [NETLIST]...[/NETLIST] will be ignored.\n\n"
        "### New Task\n"
        f"Request: {entry['spec']}\n"
        "Answer:\n"
    )

def build_prompt(entry):
    # Few-shot examples + new request
    return (
        few_shot_block() +
        "### Task\n"
        f"Request: {entry['spec']}\n"
        "Netlist:\n"
    )

def text_to_token_ids(text, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs.get("attention_mask", None)

def extract_netlist(text):
    t = text.strip()

    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()

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
        return model.generate(input_ids=input_ids, **cfg)
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **cfg)

def decode_completion_only(out_ids, prompt_input_ids, tokenizer):
    # decode only the newly generated completion - exclude prompt tokens
    gen_only = out_ids[0][prompt_input_ids.shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

def extract_netlist_from_tags(text: str) -> str:
    start_tag = "[NETLIST]"
    end_tag = "[/NETLIST]"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        body = text[start_idx + len(start_tag):end_idx]
        return body.strip()
    else:
        # Fallback: if tags are missing, just use the whole text
        # (so the evaluation still runs, even if badly).
        return text.strip()

## main runner loop (Step 2)
def main():
    torch.manual_seed(123) # 123 for reproducibility

    benchmark_path = Path("benchmark/spice_benchmark.json")
    out_dir = Path("runs/P6_format")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_parse_json(benchmark_path)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of test case objects.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model.device

    for i, entry in enumerate(data, start=1):
        case_id = entry.get("id", f"case_{i:03d}")

        input_text = build_prompt_format_control(entry)
        input_ids, attention_mask = text_to_token_ids(input_text, tokenizer, device)

        try:
            with torch.inference_mode():
                out_ids = generate(model, input_ids, attention_mask, GEN_CONFIG, tokenizer)
            completion = decode_completion_only(out_ids, input_ids, tokenizer)
            # first, strip [NETLIST]...[/NETLIST] if present
            netlist_raw = extract_netlist_from_tags(completion)
            # then optionally trim at the first .end (reuse your old helper)
            netlist_text = extract_netlist(netlist_raw)

        except Exception as e:
            netlist_text = f" - Error generating netlist for {case_id}: {e}\n.end"

        out_path = out_dir / f"{case_id}.cir" # <-- SPICE circuit file extension
        out_path.write_text(netlist_text + "\n", encoding="utf-8")

        print(f"[{i}/{len(data)}] wrote {out_path}")

    print("Complete")

if __name__ == "__main__":
    main()
