# Silvaco Device Simulation Code Dataset - Training Set

## Overview

This dataset contains **713 instruction-code pairs** for training Large Language Models to generate Silvaco device simulation code from natural language circuit design specifications.

**Task**: Given a detailed circuit design description (instruction), generate complete Silvaco SPICE netlist code (output).

**Important**: This is the **training set only**. An additional test set of 97 samples is reserved by the instructor for final evaluation.

## Dataset Files

### 1. `silvaco_dataset_train.json` (5.9MB)

Training dataset with 713 samples in instruction-following format.

**Structure**:
```json
[
  {
    "instruction": "Detailed circuit design specification...",
    "input": "",
    "output": "Complete Silvaco SPICE netlist code..."
  },
  ...
]
```

**Fields**:
- `instruction`: Natural language description of circuit design requirements (avg 1,156 characters)
- `input`: Always empty (not used)
- `output`: Generated Silvaco SPICE netlist code (avg 7,393 characters)

**Statistics**:
- Total samples: 713
- Average instruction length: ~290 words
- Average code length: ~1,850 words
- Code types: SPICE netlists, harmonic balance simulations, transient analysis

### 2. `Silvaco_Examples_Student.zip` (18MB)

Reference examples from Silvaco user manuals (optional resource).

**Contents**:
- 726 `.in` files: Silvaco input deck files
- 76 `.lib` files: Model library files
- PDF manuals: Reference documentation

**Purpose**:
- Additional context and examples for understanding Silvaco syntax and circuit patterns
- **You can use these .in files to create your own training data** by writing corresponding instruction descriptions
- Provides opportunity for data augmentation and creative exploration

## Dataset Split Recommendation

Since you have 713 training samples, recommended split:

```
Training set:   570 samples (80%)
Validation set:  143 samples (20%)
```

Or if you prefer a validation + test split for local evaluation:

```
Training set:   570 samples (80%)
Validation set:  71 samples (10%)
Local test set:  72 samples (10%)
```

**Note**: Your model will be evaluated on the instructor's hidden test set (97 samples), not your local test set.

## Data Augmentation Opportunities

### Using Reference .in Files

The provided `Silvaco_Examples_Student.zip` contains 726 additional `.in` files. You can:

1. **Analyze the circuit code** in these `.in` files
2. **Write your own instruction descriptions** explaining what the circuit does
3. **Expand your training dataset** beyond the provided 713 samples
4. **Practice understanding** Silvaco code structure and conventions

This allows you to exercise creativity while expanding your training data if desired.

## Circuit Types in Dataset

The dataset covers various RF and mixed-signal circuit designs:

- **Voltage-Controlled Oscillators (VCO)**: Phase noise, harmonic balance analysis
- **Low-Noise Amplifiers (LNA)**: S-parameters, stability, IIP3 characterization
- **Mixers**: Conversion gain, noise figure, two-tone analysis
- **Switched-Capacitor Circuits**: Transient and periodic steady-state analysis
- **Operational Amplifiers**: DC/AC analysis, stability margins
- **Power Amplifiers**: Load-pull, compression point, linearity
- **TCAD Simulations**: Process, mesh, and device simulations

## Code Format

All generated code follows Silvaco SPICE netlist format:

```spice
* Circuit Name
* Netlist Generator Information

* Schematic name: CircuitName
[Component definitions: R, C, L, M, I, V]
[Subcircuit definitions: .SUBCKT ... .ENDS]

.GLOBAL [global nodes]

* Simulation Control
.inc [model files]
.OP
.PARAM [parameters]
.OPTIONS [simulation options]
.TEMP [temperature]

* Analysis Commands
.AC / .TRAN / .HARMONIC / .HNET / .HNOISE / etc.

.END
```

## Evaluation Metrics

Your model will be evaluated on the hidden test set using these metrics:

1. **Syntax Correctness (25%)**: Can code be parsed as valid SPICE?
   - Valid SPICE syntax
   - Proper command structure
   - No syntax errors

2. **Semantic Accuracy (40%)**: Is circuit topology correct?
   - Correct component connections
   - Accurate parameter values
   - Proper analysis directives

3. **Completeness (20%)**: Is all required information present?
   - All components defined
   - Simulation commands included
   - Proper termination (.END)

4. **Code Quality (15%)**:
   - Proper formatting and indentation
   - Meaningful node names
   - Complete netlist structure

## Data Loading Example

```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load training dataset
with open('silvaco_dataset_train.json', 'r') as f:
    data = json.load(f)

# Format for training
def format_sample(sample):
    return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"

# Example usage
tokenizer = AutoTokenizer.from_pretrained("gpt2")
formatted_text = format_sample(data[0])
tokens = tokenizer(formatted_text, truncation=True, max_length=2048)
```

## Important Notes

### Model Constraint
- **Maximum model size**: ≤1B parameters
- Recommended models: GPT-2 (117M-1.5B), DistilGPT-2 (82M), TinyLlama (1.1B), T5-large (770M), Llama 3.2 (1B)

### Context Length Challenges
- Average sample length: ~8,500 characters (~2,100 tokens)
- Use models with sufficient context length (≥2048 tokens)
- Consider truncation strategies for very long samples

### Training Recommendations
- Fine-tuning method: Full fine-tuning, LoRA, or QLoRA
- Batch size: Start small (1-2) due to long sequences
- Learning rate: 5e-5 to 1e-4 (typical for fine-tuning)
- Epochs: 3-5 epochs usually sufficient
- Validation: Monitor validation loss to prevent overfitting

### Hidden Test Set
- An additional 97 samples are reserved for instructor evaluation
- These samples are NOT in your training data
- Your model will be evaluated on these unseen samples
- Focus on generalization, not memorization

## Data Quality Notes

- All code samples are from Silvaco user manuals and verified examples
- Instructions are synthetically generated to describe the circuits
- Code follows proper SPICE syntax and simulation best practices
- Some samples may reference external library files (.inc directives)

## License and Usage

- **For educational purposes only**
- Source: Silvaco user manuals and documentation
- Use for course final project
- Do not redistribute without permission

## Questions?

- See `final_project_instructions.html` for complete project details
- Ask in office hours or Slack channel #final-project
- Check Silvaco documentation in `Silvaco_Examples_Student.zip`

---

**Dataset prepared for CSC 375/575 - Generative AI Final Project**
**Prof. Rongyu Lin, Quinnipiac University, Fall 2025**
