# Educational Writing Annotator

[![Paper](https://img.shields.io/badge/arXiv-2501.11035-b31b1b)](https://arxiv.org/abs/2501.11035)
[![Main repo](https://img.shields.io/badge/GitHub-Arabic--Text--to--Crosswords-181717?logo=github)](https://github.com/KamyarZeinalipour/Arabic-Text-to-Crosswords)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97_Collection-Arabic_Crosswords-orange)](https://huggingface.co/collections/Kamyar-zeinalipour/arabic-educational-crosswords-6af46e6797758d5046eb4f39)

The annotation tool and expert ratings used in the human evaluation of
[*From Arabic Text to Puzzles: LLM-Driven Development of Arabic Educational Crosswords*](https://arxiv.org/abs/2501.11035).
The `Annotated/` folder holds the ratings (A–E) for clues written by GPT-4-Turbo, GPT-3.5-Turbo
(base and fine-tuned) and Llama-3-8B-Instruct (base and fine-tuned,
[Llama3-8B-Ar-Text-to-Cross](https://huggingface.co/Kamyar-zeinalipour/Llama3-8B-Ar-Text-to-Cross))
on 200 test passages.

## Overview
This script provides a tool for annotating educational writing samples with categorization and additional metadata.

## Prerequisites
- Python installed on your system
- Required CSV input file

## Installation
1. Clone the repository
2. Ensure you have the necessary dependencies installed

## Usage

### Command Syntax
```bash
python educw_annotation.py --examples_batch_folder <path-to-csv-file>.csv --annotator_name <Annotator Name>
```

### Parameters
- `--examples_batch_folder`: Path to the input CSV file
- `--annotator_name`: Name of the annotator (for tracking purposes)

## Input File Requirements

Your input CSV must contain the following mandatory columns:
- `extract`: The text to be annotated
- `category`: Classification or type of text
- `clue`: Contextual information or hints
- `answer`: Keywords or specific answer related to the text

## Output

### Directory Structure
- A new `annotations` folder will be created in the main directory
- Annotated files will be saved as `annotations_{file_name}.csv`

### Annotated Files Location
- Original annotated files are stored in the `Annotated` folder

## Example Command
```bash
python educw_annotation.py --examples_batch_folder educational_samples.csv --annotator_name "Research Team"
```

## Troubleshooting
- Ensure the input CSV matches the required column structure
- Check that the file path is correct
- Verify Python and required libraries are installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

---

Questions: please open an issue.
