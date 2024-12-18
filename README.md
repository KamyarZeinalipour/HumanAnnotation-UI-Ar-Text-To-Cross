# Educational Writing Annotator

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

Feel free to reach out if you have any questions or need assistance with the setup. Happy clue generating!
