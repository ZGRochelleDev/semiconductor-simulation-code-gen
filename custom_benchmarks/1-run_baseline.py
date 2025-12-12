## Final Project: Semiconductor Simulation Code Generation
## CSC 575 - 02_25FA - Generative A.I. - Dr. Lin
## Zoe Rochelle
## 12/03/2025

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

## avoid generate() warnings if pad token isnt set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

## basic prompt ##
# note:
    # ngspice is a subset of SPICE and is (hopefully) easier for the model to generate correctly because it's popular
    # source: https://ngspice.sourceforge.io/ngspice-tutorial.html
prompt_1 = (
    "You are generating an ngspice-compatible netlist.\n"
    "I'd like you to only return the netlist text (no explanation, no markdown).\n\n"
    "Task: Create a SPICE netlist for an RC low-pass filter with R = 1k and C = 100nF."
)

## this prompt defines some requirements ##
# source: https://youtu.be/4ew1GI_b23M?si=_WAib_yMCI6LhDlt
#    - the document on the right is basically what we are tyring to get the model to produce
prompt_2 = (
"Please write an ngspice-compatible SPICE netlist and output only the netlist.\n"
"Include a voltage source, the R and C elements, an AC analysis command, and .end.\n\n"

"Circuit: RC low-pass filter with R=1k, C=100nF."
)

## this prompt adds further explicit instructions: uses node names and topology ##
# source: https://ngspice.sourceforge.io/ngspice-tutorial.html
# note:
    # resistor is between IN --> OUT
    # capacitor is between OUT --> ground (0)
prompt_3 = (
    "Please produce a valid ngspice netlist only (no prose).\n" # <-- only output code
    "Use node 0 as ground. Use nodes IN and OUT.\n\n"
    "Build an RC low-pass filter with R=1k from IN to OUT and C=100nF from OUT to 0."
)

## final prompt: modified version of prompt_2, which gave me the least bad results ##
# note: no prose might not have any impact when using Qwen3-0.6B-Base
prompt_4 = (
    "Write an ngspice-compatible SPICE netlist.\n"
    "Output ONLY the netlist text (no explanation, no markdown).\n"
    "Must include: a voltage source, R and C elements, an analysis command, and .end.\n"
    "Use node 0 as ground. Use nodes IN and OUT.\n\n"
    "Build an RC low-pass filter with R=1k from IN to OUT and C=100nF from OUT to 0.\n"
    "Use an AC source of 1V and run: .ac dec 100 10 1Meg\n"
    "The final line must be exactly: .end\n"
)

inputs = tokenizer(prompt_4, return_tensors="pt").to(model.device)

gen_cfg = dict(
    do_sample=False,
    max_new_tokens=200,
    pad_token_id=tokenizer.pad_token_id,
    repetition_penalty=1.1,
    no_repeat_ngram_size=6, # kept getting repeated lines w/o this
    eos_token_id=tokenizer.eos_token_id
)

with torch.inference_mode():
    out_ids = model.generate(**inputs, **gen_cfg)

gen_only = out_ids[0][inputs["input_ids"].shape[1]:] # Decode only the newly generated completion
netlist_text = tokenizer.decode(gen_only, skip_special_tokens=True).strip() # to avoid failing exact-match checks from trailing whitespace


## the base models is outputting markdown so I'm cleaning that up here ##
text = netlist_text

if "```" in text:
    parts = text.split("```")
    if len(parts) >= 3:
        text = parts[1].strip()

lower = text.lower()
end_idx = lower.find(".end")
if end_idx != -1:
    text = text[: end_idx + len(".end")].strip()

netlist_text = text

## output results ##
print("Complete")
with open('baseline-output.txt', 'w') as file:
    file.write(netlist_text)
print("Content written to baseline-output.txt")
