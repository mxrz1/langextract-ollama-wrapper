# LangExtract NER Benchmark

Benchmark Ollama LLMs on cybersecurity named entity recognition (NER) datasets using [LangExtract](https://github.com/google/langextract). Tracks extraction accuracy, energy consumption, and carbon emissions via [CodeCarbon](https://codecarbon.io/). It provides the possibility to define entity types descriptions and implements a simple greedy algorithm (sentences with the most entity types missing first) to select as few as possible example sentences to not overwhelm smaller LLMs with large prompts. 

## Overview

The benchmark runs a configurable Ollama model against one or more cybersecurity NER datasets, measures prediction quality against ground-truth BIO annotations, and records per-run energy and emissions data. Results are stored in structured output files ready for downstream evaluation.

**Key capabilities:**
- Few-shot NER via LangExtract with Ollama (no fine-tuning required)
- Sentence-level and document-level extraction modes
- Automatic energy and CO₂ tracking via CodeCarbon
- One-command setup and execution with `run.sh`

---

## Requirements

| Requirement | Notes |
|---|---|
| **Linux** | Ubuntu/Debian recommended; `run.sh` uses `apt` for setup |
| **Python 3.12** | Installed automatically by `run.sh` if missing |
| **Ollama** | Installed automatically by `run.sh` if missing |
| **NVIDIA GPU** | Optional but strongly recommended for reasonable inference speed |
| **`curl`, `zip`** | Standard utilities, pre-installed on most systems |
| **Internet access** | For initial Ollama and model downloads |

---

## Quick Start (Automated)

`run.sh` handles environment setup, dependency installation, Ollama installation, model downloads, extraction runs, and results archiving in one command.

```bash
# Make executable (first time only)
chmod +x run.sh

# Run a single config
./run.sh configs/stixnet_single_sentence_gemma3-27b_config.yaml

# Run all configs
./run.sh configs/*.yaml
```

The script will:
1. Create a Python 3.12 virtual environment (`venv312/`)
2. Install Python dependencies from `requirements.txt`
3. Install Ollama and start the service
4. Pull any required models not yet downloaded
5. Run `run_extraction.py` for each config sequentially
6. Archive all results to `results_YYYYMMDD_HHMMSS.zip`

---

## Manual Setup

If you prefer to set up and run manually:

```bash
# 1. Create and activate virtual environment
python3.12 -m venv venv312
source venv312/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (in a separate terminal or background)
ollama serve &

# 4. Pull the model you want to use
ollama pull gemma3:27b

# 5. Run extraction
python run_extraction.py --config configs/stixnet_single_sentence_gemma3-27b_config.yaml
```

---

## Configuration

Each experiment is defined by a YAML config file. Example:

```yaml
# Model settings
model:
  model_id: "gemma3:27b"           # Ollama model identifier
  model_url: "http://localhost:11434"  # Ollama API URL
  max_retries: 5                    # Retries per sentence on extraction failure
  num_ctx: 4096                     # Ollama context window in tokens 
  max_char_buffer: 1200             # Max characters per LangExtract chunk 

# Dataset settings
dataset:
  dataset_dir: "datasets/stixnet_single_sentence"  # Dataset to run extraction on
  splits: ["test"]                  # Splits to process: "train", "valid", "test"
  train_file: null                  # Path to training data for few-shot examples; null = auto-detect from dataset_dir

# Extraction settings
extraction:
  num_examples: 1                   # Few-shot examples per entity type (higher = larger prompt)
  prompt: |
    Extract cybersecurity threat intelligence entities from the text.
  entity_types:                     # Entity types to extract (keys = type names)
    intrusion-set:
      description: "A grouped set of adversarial behaviors..."
    malware:
      description: "Malicious code inserted into a system..."
    # ... additional entity types

# Output settings
output:
  output_dir: null       # null = auto-generate as results/{dataset}/{model}/
  save_jsonl: true       # Save LangExtract JSONL format
  save_predictions: true # Save BIO prediction format (token GT PRED)
```

### Key Parameter Notes

**`num_ctx`** - Ollama context window (tokens). 

**`max_char_buffer`** - LangExtract document chunking (characters).

**`train_file`** - Path to BIO-tagged training data file to be used to build few-shot examples. For whole-document datasets, point this to the single-sentence train split to keep prompt size manageable:
```yaml
dataset:
  dataset_dir: "datasets/stixnet_whole_document"
  train_file: "datasets/stixnet_single_sentence/train.txt"
```

**`num_examples`** - Number of entity mentions per type to include as few-shot examples. 

---

## Available Configs

| Config file | Dataset | Model | Notes                                    |
|---|---|---|------------------------------------------|
| `stixnet_single_sentence_gemma3-27b_config.yaml` | stixnet_single_sentence | gemma3:27b | Sentence-level, 16 STIX entity types     |
| `stixnet_single_sentence_deepseek-r1-32b_config.yaml` | stixnet_single_sentence | deepseek-r1:32b | Sentence-level, 16 STIX entity types     |
| `stixnet_whole_document_gemma3-27b_config.yaml` | stixnet_whole_document | gemma3:27b | Document-level, single-sentence examples |
| `stixnet_whole_document_deepseek-r1-32b_config.yaml` | stixnet_whole_document | deepseek-r1:32b | Document-level, single-sentence examples |
| `dnrti_gemma3-27b_config.yaml` | dnrti | gemma3:27b | 13 entity types                          |
| `dnrti_deepseek-r1-32b_config.yaml` | dnrti | deepseek-r1:32b | 13 entity types                          |

---

## Datasets

All datasets use BIO (Beginning-Inside-Outside) tagging format and are located in `datasets/`.

```
APT3    B-intrusion-set
used    O
RDP     B-tool
```

### stixnet_single_sentence

Individual sentences from STIX cybersecurity threat intelligence reports. 16 entity types based on the STIX 2.1 standard:

`intrusion-set`, `attack-pattern`, `malware`, `tool`, `identity`, `location`, `vulnerability`, `campaign`, `threat-actor`, `domain-name`, `url`, `indicator`, `file_paths`, `sha256s`, `tactic`

### stixnet_whole_document

stixnet_single_sentence, but the documents not split into sentences. 

### dnrti

Threat intelligence reports annotated with 13 domain-specific entity types:

`HackOrg`, `OffAct`, `SamFile`, `SecTeam`, `Tool`, `Time`, `Purp`, `Area`, `Idus`, `Org`, `Way`, `Exp`, `Features`

Each dataset contains `train.txt`, `valid.txt`, `test.txt`, and `all.txt` splits.

---

## Output Files

Results are written to `results/{dataset_name}/{model_name}/` by default.

| File | Description |
|---|---|
| `{split}_predictions.txt` | Three-column BIO file: `token  ground_truth  prediction` |
| `{split}_extractions.jsonl` | LangExtract JSON format with character-level positions |
| `emissions.csv` | CodeCarbon energy and emissions data per run |
| `results_YYYYMMDD_HHMMSS.zip` | Archive of entire results folder (created by `run.sh`) |

### emissions.csv columns

| Column | Description |
|---|---|
| `duration` | Run duration in seconds |
| `emissions` | CO₂ equivalent in kg |
| `energy_consumed` | Total energy in kWh |
| `cpu_power` / `gpu_power` / `ram_power` | Power in Watts |
| `cpu_energy` / `gpu_energy` / `ram_energy` | Energy in kWh |
| `gpu_model`, `cpu_model` | Hardware identifiers |


## Adding a New Model or Config

1. Pull the model in Ollama:
   ```bash
   ollama pull llama3.2:3b
   ```

2. Copy an existing config and adjust:
   ```bash
   cp configs/stixnet_single_sentence_gemma3-27b_config.yaml \
      configs/stixnet_single_sentence_llama3.2-3b_config.yaml
   ```

3. Edit the new config and update `model_id`:
   ```yaml
   model:
     model_id: "llama3.2:3b"
   ```

4. Run:
   ```bash
   python run_extraction.py --config configs/stixnet_single_sentence_llama3.2-3b_config.yaml
   ```
