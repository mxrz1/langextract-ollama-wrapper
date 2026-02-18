#!/usr/bin/env python3
"""
Entity extraction script using LangExtract with Ollama models.
Saves predictions.txt, emissions.csv, JSONL, and HTML visualizations.

Usage:
    python run_extraction.py --config configs/stixnet_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple
from datetime import datetime

import yaml
import langextract as lx
from codecarbon import EmissionsTracker


class Sentence:
    """Represents a sentence with tokens and BIO tags."""
    def __init__(self, tokens: List[str], tags: List[str]):
        self.tokens = tokens
        self.tags = tags
        self.text = ' '.join(tokens)

    def __len__(self):
        return len(self.tokens)


class ExtractionConfig:
    """Configuration for entity extraction loaded from YAML."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Model settings
        self.model_id = self.config['model']['model_id']
        self.model_url = self.config['model']['model_url']
        self.max_retries = self.config['model'].get('max_retries', 3)
        self.num_ctx = self.config['model'].get('num_ctx', 2048)
        self.max_char_buffer = self.config['model'].get('max_char_buffer', 1000)

        # Dataset settings
        self.dataset_dir = self.config['dataset'].get('dataset_dir')
        self.dataset_file = self.config['dataset'].get('dataset_file')
        self.splits = self.config['dataset'].get('splits', ['train', 'valid', 'test'])
        self.train_file = self.config['dataset'].get('train_file')

        # Extraction settings
        self.num_examples = self.config['extraction']['num_examples']
        self.prompt_template = self.config['extraction']['prompt']
        self.entity_types = self.config['extraction']['entity_types']

        # Output settings
        self.output_dir = self.config['output'].get('output_dir')
        self.save_jsonl = self.config['output'].get('save_jsonl', True)
        self.save_predictions = self.config['output'].get('save_predictions', True)

    def create_prompt(self, entity_types_in_data: Set[str]) -> str:
        """Create extraction prompt with entity descriptions if available."""
        types_to_use = entity_types_in_data if entity_types_in_data else set(self.entity_types.keys())

        # Check if any descriptions exist
        has_descriptions = False
        entity_list_parts = []
        for entity_type in sorted(types_to_use):
            if entity_type in self.entity_types:
                desc = self.entity_types[entity_type].get('description', '') if isinstance(self.entity_types[entity_type], dict) else ''
                if desc:
                    has_descriptions = True
                    entity_list_parts.append(f"  - {entity_type}: {desc}")
                else:
                    entity_list_parts.append(f"  - {entity_type}")
            else:
                entity_list_parts.append(f"  - {entity_type}")

        # Only add entity list if we have descriptions
        if has_descriptions:
            entity_list = "\n".join(entity_list_parts)
            prompt = f"{self.prompt_template.strip()}\n\nEntity types to extract:\n{entity_list}"
        else:
            prompt = self.prompt_template.strip()

        return prompt


def load_bio_data(file_path: str) -> Tuple[List[Sentence], Set[str]]:
    """Load BIO-tagged data from file."""
    sentences = []
    entity_types = set()
    current_tokens = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(Sentence(current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue

            if "\t" in line:
                parts = line.split('\t')
            else:
                parts = line.split(" ")
            if len(parts) == 2:
                token, tag = parts
                current_tokens.append(token)
                current_tags.append(tag)

                if tag.startswith('B-') or tag.startswith('I-'):
                    entity_type = tag[2:]
                    entity_types.add(entity_type)

        if current_tokens:
            sentences.append(Sentence(current_tokens, current_tags))

    return sentences, entity_types


def create_few_shot_examples_from_train(train_file: str, num_examples: int,
                                        expected_entity_types: Set[str] = None) -> List[lx.data.ExampleData]:
    """Create few-shot examples from training data using hybrid budget approach.

    Args:
        train_file: Path to training data file
        num_examples: Number of entity MENTIONS per entity type (not sentences)
        expected_entity_types: Optional set of entity types that should be covered

    Returns:
        List of unique ExampleData objects with minimal duplication

    Algorithm:
        - Deduplicates sentences by text
        - Sorts by number of entity types (prioritizes multi-entity sentences)
        - Uses budget tracking: each entity type needs num_examples mentions
        - Selects sentences that contribute to under-represented entity types
        - Stops when all budgets satisfied or no more helpful sentences
    """
    if not os.path.exists(train_file):
        print(f"WARNING: {train_file} not found")
        return []

    sentences, entity_types_in_train = load_bio_data(train_file)

    # Parse all sentences with their entity types
    sentences_with_entities = []
    for sentence in sentences:
        sentence_entity_types = set()
        for tag in sentence.tags:
            if tag.startswith('B-'):
                entity_type = tag[2:]
                sentence_entity_types.add(entity_type)

        if sentence_entity_types:  # Only include sentences with entities
            sentences_with_entities.append((sentence, sentence_entity_types))

    # Deduplicate by sentence text
    unique_sentences = {}  # sentence_text -> (sentence_obj, set_of_entity_types)
    for sentence, entity_types in sentences_with_entities:
        sent_text = sentence.text
        if sent_text not in unique_sentences:
            unique_sentences[sent_text] = (sentence, entity_types)

    # Sort by number of entity types (descending) - prioritize multi-entity sentences
    sorted_sentences = sorted(
        unique_sentences.values(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Initialize budgets: each entity type needs num_examples mentions
    budgets = {et: num_examples for et in entity_types_in_train}

    # Select sentences using budget tracking
    selected_sentences = []
    mention_counts = {et: 0 for et in entity_types_in_train}

    for sentence, entity_types in sorted_sentences:
        # Check if this sentence helps any entity type with remaining budget
        helps_any = any(budgets.get(et, 0) > 0 for et in entity_types)

        if helps_any:
            selected_sentences.append(sentence)

            # Decrease budget for all entity types in this sentence
            for et in entity_types:
                if et in budgets and budgets[et] > 0:
                    budgets[et] -= 1
                    mention_counts[et] += 1

        # Early exit if all budgets satisfied
        if all(b == 0 for b in budgets.values()):
            break

    # Convert sentences to LangExtract examples
    examples = []
    for sentence in selected_sentences:
        extractions = []
        current_entity = None
        entity_tokens = []

        for token, tag in zip(sentence.tokens, sentence.tags):
            if tag.startswith('B-'):
                if current_entity:
                    extractions.append(lx.data.Extraction(
                        extraction_class=current_entity,
                        extraction_text=' '.join(entity_tokens),
                        attributes={}
                    ))
                current_entity = tag[2:]
                entity_tokens = [token]

            elif tag.startswith('I-') and current_entity:
                entity_tokens.append(token)

            elif tag == 'O':
                if current_entity:
                    extractions.append(lx.data.Extraction(
                        extraction_class=current_entity,
                        extraction_text=' '.join(entity_tokens),
                        attributes={}
                    ))
                    current_entity = None
                    entity_tokens = []

        if current_entity:
            extractions.append(lx.data.Extraction(
                extraction_class=current_entity,
                extraction_text=' '.join(entity_tokens),
                attributes={}
            ))

        if extractions:
            examples.append(lx.data.ExampleData(text=sentence.text, extractions=extractions))

    # Print coverage statistics
    print(f"Created {len(examples)} few-shot examples (target: {num_examples} mentions per entity type)")
    print(f"  Total unique sentences: {len(unique_sentences)}")
    print(f"  Selected for examples: {len(selected_sentences)}")

    # Show per-entity-type mention counts
    print(f"\nEntity type mention coverage:")
    for entity_type in sorted(mention_counts.keys()):
        count = mention_counts[entity_type]
        status = "✓" if count >= num_examples else f"⚠ {count}/{num_examples}"
        print(f"  {status} {entity_type}")

    # Check for missing entity types
    if expected_entity_types:
        covered_types = set(et for et, count in mention_counts.items() if count > 0)
        missing_types = expected_entity_types - covered_types

        if missing_types:
            print(f"\n⚠️  WARNING: The following entity types are NOT in the few-shot examples:")
            print(f"   Missing types: {sorted(missing_types)}")
            print(f"   This may impact extraction quality, especially for Gemini.")
            print(f"   For Ollama: Model can still extract these if in prompt descriptions.")
            print(f"   For Gemini: Model CANNOT extract these with use_schema_constraints=True.")

        print(f"\n✓ Entity type coverage: {len(covered_types)}/{len(expected_entity_types)} types")
        print(f"  Covered: {sorted(covered_types)}")
        if missing_types:
            print(f"  Missing: {sorted(missing_types)}")

    return examples


def extract_from_sentence(sentence: Sentence, prompt: str, examples: List, model_id: str,
                          model_url: str, max_retries: int = 3, num_ctx: int = 2048,
                          max_char_buffer: int = 1000):
    """Extract entities from a single sentence and return (predictions, result_document, success)."""
    predictions = ["O"] * len(sentence.tokens)
    result_doc = None

    for attempt in range(max_retries):
        try:
            result = lx.extract(
                text_or_documents=sentence.text,
                prompt_description=prompt,
                examples=examples,
                model_id=model_id,
                model_url=model_url,
                fence_output=False,
                use_schema_constraints=True,
                show_progress=False,
                max_char_buffer=max_char_buffer,
                language_model_params={"num_ctx": num_ctx},
            )

            if isinstance(result, list):
                if len(result) == 0:
                    return predictions, None, False
                result = result[0]

            result_doc = result  # Save for JSONL

            if not hasattr(result, 'extractions'):
                if attempt < max_retries - 1:
                    print(f"WARNING: No extractions. Retrying...")
                    continue
                return predictions, result_doc, False

            # Map extractions to tokens
            text = result.text
            token_positions = []
            current_pos = 0

            for token in sentence.tokens:
                start_idx = text.find(token, current_pos)
                if start_idx != -1:
                    end_idx = start_idx + len(token)
                    token_positions.append((start_idx, end_idx))
                    current_pos = end_idx
                else:
                    token_positions.append((None, None))

            # Align extractions to tokens
            for extraction in result.extractions:
                if extraction.char_interval is None or extraction.char_interval.start_pos is None:
                    continue

                extraction_start = extraction.char_interval.start_pos
                extraction_end = extraction.char_interval.end_pos or (extraction_start + len(extraction.extraction_text))
                extraction_class = extraction.extraction_class

                for idx, (token_start, token_end) in enumerate(token_positions):
                    if token_start is None:
                        continue

                    # Check if token overlaps with extraction
                    if token_start < extraction_end and token_end > extraction_start:
                        # Determine if this is the FIRST token of this extraction
                        # (i.e., token contains the extraction start position)
                        is_first_token = (token_start <= extraction_start < token_end)

                        if is_first_token and predictions[idx] == "O":
                            predictions[idx] = f"B-{extraction_class}"
                        elif predictions[idx] == "O":
                            predictions[idx] = f"I-{extraction_class}"

            return predictions, result_doc, True  # Success!

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"ERROR (attempt {attempt + 1}): {e}")
            else:
                print(f"ERROR after {max_retries} attempts: {e}")
                return predictions, None, False

    return predictions, result_doc, False


def process_file(file_path: str, config: ExtractionConfig, output_dir: str):
    """Process a single dataset file with CodeCarbon tracking."""
    split_name = Path(file_path).stem

    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")

    # Load data
    sentences, entity_types = load_bio_data(file_path)
    print(f"Loaded {len(sentences)} sentences")
    print(f"Entity types: {sorted(entity_types)}")

    # Create prompt
    prompt = config.create_prompt(entity_types)

    # Get training examples
    train_file = config.train_file
    if train_file is None:
        potential_train = os.path.join(config.dataset_dir or os.path.dirname(file_path), "train.txt")
        if os.path.exists(potential_train):
            train_file = potential_train

    examples = []
    if train_file and os.path.exists(train_file):
        print(f"\nCreating {config.num_examples} few-shot examples from: {train_file}")
        expected_types = set(config.entity_types.keys())
        examples = create_few_shot_examples_from_train(train_file, config.num_examples, expected_types)

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"langextract_{config.model_id.replace(':', '_')}_{split_name}",
        tracking_mode="machine",
        measure_power_secs=15,
        save_to_file=True,
        output_dir=output_dir,
        output_file="emissions.csv",
        on_csv_write="update"  # Update existing row instead of append
    )

    # Start tracking
    print(f"\n[{split_name.upper()}] Starting extraction with CodeCarbon tracking...")
    tracker.start()
    start_time = datetime.now()

    # Process sentences
    all_ground_truth = []
    all_predictions = []
    extraction_results = []  # For JSONL
    successful_extractions = 0
    failed_extractions = 0

    for idx, sentence in enumerate(sentences, 1):
        print(f"[{split_name.upper()}] Processing {idx}/{len(sentences)}...", end=" ", flush=True)

        predictions, result_doc, success = extract_from_sentence(
            sentence, prompt, examples, config.model_id, config.model_url,
            config.max_retries, config.num_ctx, config.max_char_buffer
        )

        all_ground_truth.extend(sentence.tags)
        all_predictions.extend(predictions)

        if success:
            successful_extractions += 1
            print("✓")  # Print success with newline
        else:
            failed_extractions += 1
            print("✗")  # Print failure with newline

        if result_doc is not None:
            extraction_results.append(result_doc)

    # Stop tracking
    end_time = datetime.now()
    emissions_data = tracker.stop()
    duration = (end_time - start_time).total_seconds()

    print(f"[{split_name.upper()}] Completed in {duration:.2f} seconds")
    if emissions_data is not None:
        print(f"[{split_name.upper()}] Emissions: {emissions_data:.6e} kgCO2eq")
    else:
        print(f"[{split_name.upper()}] Emissions: Not tracked (codecarbon conflict)")

    # Save predictions.txt
    if config.save_predictions:
        predictions_file = os.path.join(output_dir, f"{split_name}_predictions.txt")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            pred_idx = 0
            for sentence in sentences:
                for token, gt_tag in zip(sentence.tokens, sentence.tags):
                    pred_tag = all_predictions[pred_idx]
                    f.write(f"{token} {gt_tag} {pred_tag}\n")
                    pred_idx += 1
                f.write("\n")
        print(f"✓ Saved predictions: {predictions_file}")

    # Save JSONL
    if config.save_jsonl and extraction_results:
        jsonl_file = os.path.join(output_dir, f"{split_name}_extractions.jsonl")

        # Save using LangExtract's function - include .jsonl in the name
        lx.io.save_annotated_documents(
            extraction_results,
            output_name=f"{split_name}_extractions.jsonl",  # Include .jsonl extension
            output_dir=output_dir
        )

        print(f"✓ Saved JSONL: {jsonl_file}")

    # Print summary
    num_predicted = sum(1 for tag in all_predictions if tag.startswith('B-'))
    num_gt = sum(1 for tag in all_ground_truth if tag.startswith('B-'))
    total_sentences = len(sentences)
    success_rate = (successful_extractions / total_sentences * 100) if total_sentences > 0 else 0

    print(f"\n[{split_name.upper()}] Summary:")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Successful extractions: {successful_extractions} ({success_rate:.1f}%)")
    print(f"  Failed extractions: {failed_extractions} ({100-success_rate:.1f}%)")
    print(f"  Ground truth entities: {num_gt}")
    print(f"  Predicted entities: {num_predicted}")


def main():
    parser = argparse.ArgumentParser(description="Extract entities with JSONL/HTML output")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--dataset-dir", type=str, help="Override dataset directory")
    parser.add_argument("--splits", type=str, nargs="+", help="Override splits")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    config = ExtractionConfig(args.config)

    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    if args.splits:
        config.splits = args.splits

    # Determine files to process
    files_to_process = []
    if config.dataset_dir:
        if not os.path.isdir(config.dataset_dir):
            print(f"ERROR: Directory not found: {config.dataset_dir}")
            sys.exit(1)

        for split_name in config.splits:
            file_path = os.path.join(config.dataset_dir, f"{split_name}.txt")
            if os.path.exists(file_path):
                files_to_process.append(file_path)
            else:
                print(f"WARNING: {file_path} not found, skipping")
    else:
        print("ERROR: No dataset_dir in config")
        sys.exit(1)

    if not files_to_process:
        print("ERROR: No files to process")
        sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir or config.output_dir
    if output_dir is None:
        dataset_name = os.path.basename(os.path.normpath(config.dataset_dir))
        model_name = config.model_id.replace(":", "_").replace("/", "_")
        output_dir = os.path.join("results", dataset_name, model_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Model: {config.model_id}")

    # Process each file
    for file_path in files_to_process:
        try:
            process_file(file_path, config, output_dir)
        except Exception as e:
            print(f"\nERROR processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Results: {output_dir}")
    print(f"  - {split_name}_predictions.txt (BIO format)")
    print(f"  - {split_name}_extractions.jsonl (LangExtract format)")
    print(f"  - emissions.csv (CodeCarbon)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()