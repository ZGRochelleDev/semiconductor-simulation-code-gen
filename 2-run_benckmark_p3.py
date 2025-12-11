# Prompt Engineering Experiments — Step 2 (P3 Runner)
# P3: Critique -> Revise (2-pass)
#    Pass 1: draft netlist
#    Pass 2: checklist-based revision

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
    # the same 3 examples as P2 (OP / TRAN / AC)
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

        "Revision rules (must follow):\n"
        "- Do NOT rename any existing nodes, components, or sources.\n"
        "- Do NOT change numeric values unless required to satisfy the checklist.\n"
        "- Preserve correct lines exactly as-is.\n"
        "- Only ADD missing lines or minimally EDIT incorrect lines.\n"
        "- Ensure the final line is exactly: .end\n\n"
    )

def build_prompt_pass1(entry):
    # draft generation prompt, few-shot + task
    return (
        few_shot_block() +
        "### Task\n"
        f"Request: {entry['spec']}\n"
        "Netlist:\n"
    )

def build_checklist_text(entry):
    # i'm getting this checklist comes from benchmark requirements
    must = entry.get("must_contain", [])
    any_groups = entry.get("must_contain_any", [])

    lines = []
    lines.append("Checklist (the revised netlist must satisfy all items):")
    lines.append("- Must be an ngspice-compatible SPICE netlist.")
    lines.append("- Output only the netlist text (no explanation, no markdown).")
    lines.append("- Use node 0 as ground.")
    lines.append("- Final line must be exactly: .end")

    if must:
        lines.append("- Must contain these required substrings (case-insensitive):")
        for s in must:
            lines.append(f"  - {s}")

    if any_groups:
        lines.append("- Must satisfy these 'at least one' requirements (case-insensitive):")
        for group in any_groups:
            joined = " OR ".join(group)
            lines.append(f"  - Include at least one of: {joined}")

    return "\n".join(lines)

## changed this pass, this will prevent the behavior of lines losing their element
def build_prompt_pass2(entry, draft_netlist):
    checklist = build_checklist_text(entry)

    return (
        "You are an expert ngspice netlist reviewer.\n"
        "Your job is to PATCH the draft netlist so it satisfies the checklist.\n"
        "Return ONLY the corrected netlist text (no explanation, no markdown, no code fences).\n\n"
        "Revision rules (must follow):\n"
        "- Do NOT rename any nodes, components, sources, or models.\n"
        "- Do NOT remove element designators (V1, R1, C1, L1, M1, etc.).\n"
        "- Do NOT change numeric values unless required by the checklist.\n"
        "- Preserve correct lines exactly as-is.\n"
        "- Only add missing lines or minimally edit incorrect lines.\n"
        "- Ensure the final line is exactly: .end\n\n"
        f"Request:\n{entry['spec']}\n\n"
        f"Draft netlist:\n{draft_netlist}\n\n"
        f"{checklist}\n\n"
        "Corrected netlist:\n"
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

    lower = t.lower()
    end_idx = lower.rfind(".end")   # <-- last occurrence!
    if end_idx != -1:
        t = t[: end_idx + len(".end")].strip()

    return t

def generate(model, input_ids, attention_mask, gen_config, tokenizer):
    cfg = dict(gen_config)
    cfg["pad_token_id"] = tokenizer.pad_token_id
    cfg["eos_token_id"] = tokenizer.eos_token_id

    if attention_mask is None:
        return model.generate(input_ids=input_ids, **cfg)
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **cfg)


def decode_completion_only(out_ids, prompt_input_ids, tokenizer):
    gen_only = out_ids[0][prompt_input_ids.shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

def run_one_pass(model, tokenizer, device, prompt_text):
    input_ids, attention_mask = text_to_token_ids(prompt_text, tokenizer, device)
    with torch.inference_mode():
        out_ids = generate(model, input_ids, attention_mask, GEN_CONFIG, tokenizer)
    completion = decode_completion_only(out_ids, input_ids, tokenizer)
    return extract_netlist(completion)

def normalize_text(text):
    return " ".join(text.lower().split())

def quick_coverage(entry, netlist_text):
    t = normalize_text(netlist_text)

    must = entry.get("must_contain", [])
    any_groups = entry.get("must_contain_any", [])

    found = 0
    total = 0

    for s in must:
        total += 1
        if normalize_text(s) in t:
            found += 1

    for group in any_groups:
        total += 1
        if any(normalize_text(opt) in t for opt in group):
            found += 1

    return (found / total) if total else 0.0

def main():
    torch.manual_seed(123) # 123 for reproducibility

    benchmark_path = Path("benchmark/spice_benchmark.json")
    out_dir = Path("runs/P3")
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

        try:
            # pass 1: draft
            p1 = build_prompt_pass1(entry)
            draft = run_one_pass(model, tokenizer, device, p1)

            draft_cov = quick_coverage(entry, draft)

            # keep the draft if it satisfies most requirements
            if draft_cov >= 0.85:
                netlist_text = draft
            else:
                # pass 2: revise based on checklist
                p2 = build_prompt_pass2(entry, draft)
                revised = run_one_pass(model, tokenizer, device, p2)

                # keep whichever is better according to the coverage
                revised_cov = quick_coverage(entry, revised)
                netlist_text = revised if revised_cov >= draft_cov else draft

        except Exception as e:
            netlist_text = f" - Error generating netlist for {case_id}: {e}\n.end"

        out_path = out_dir / f"{case_id}.cir"
        out_path.write_text(netlist_text + "\n", encoding="utf-8")

        print(f"[{i}/{len(data)}] wrote {out_path}")

    print("Complete")

if __name__ == "__main__":
    main()
