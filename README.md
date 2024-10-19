# BERT Fine-Tuning for Phishing URL Detection

This project fine-tunes a BERT model for phishing URL detection using the Hugging Face Transformers library.

## Project Structure

- `main.py`: Main script for loading data, fine-tuning the BERT model, and evaluating its performance.
- `requirements.txt`: List of Python dependencies required for the project.
- `bert_fine_tunning.ipynb`: Jupyter notebook version of the fine-tuning process (if applicable).

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/imanoop7/Fine-Tuning-BERT-Model-for-Text-Classification
   cd Fine-Tuning-BERT-Model-for-Text-Classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

   Fine-tune the BERT model:
   ```
   python main.py
   ```

## Dataset

The project uses a custom dataset for phishing URL classification, available at `imanoop7/phishing_url_classification` on the Hugging Face Hub.

## Model

The base model used is `bert-base-uncased`, which is fine-tuned for binary classification of URLs as safe or potentially phishing.

## Results

After fine-tuning, the model's performance can be evaluated using accuracy and AUC metrics. Refer to the output of `main.py` for detailed results.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
