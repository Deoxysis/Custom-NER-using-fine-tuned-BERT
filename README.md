# Domain-Specific NER Fine-Tuning with BERT

This repository contains a collection of Jupyter Notebooks demonstrating how to fine-tune a BERT model (`bert-base-uncased`) for Named Entity Recognition (NER) across three distinct domains: **General English**, **Medical**, and **Legal**. 

Built using PyTorch and Hugging Face Transformers, each notebook implements an end-to-end token classification pipeline. The core architecture and training loop remain consistent across domains, with the primary differences isolated to dataset preprocessing logic and domain-specific entity schemas.

##  Features & Implementation Details

Each notebook follows a robust, standardized pipeline for NER:

* **Data Preprocessing:** Loads domain-specific BIO/IOB annotations, normalizes them into sentence-level examples, and builds dynamic label-to-index dictionaries.
* **Advanced Tokenization:** Utilizes `BertTokenizerFast` with offset mappings. This ensures word-level tags are correctly aligned to WordPiece tokens. Crucially, it labels only the first subtoken of a word and masks padding/special positions (assigning `-100`) so they are ignored during loss calculation.
* **Custom Data Handling:** Implements a custom PyTorch `Dataset` and `DataLoader` utilizing an 80/20 train-test split, configured for small batches (e.g., batch size of 4) with max-length truncation and padding.
* **Model Architecture:** Uses `BertForTokenClassification` initialized from `bert-base-uncased`, dynamically sizing the classification head to match the specific domain's label set.
* **Training Loop:** Optimized for GPU (when available) using the AdamW optimizer over multiple epochs.
* **Rigorous Evaluation:** Evaluates held-out test data using both raw token accuracy and the `seqeval` library. The `seqeval` classification report (Precision, Recall, F1-score) is used to accurately measure entity-level performance and account for the severe class imbalance caused by the dominant `O` (Outside) tag.

##  Repository Structure

* `General_NER.ipynb`: Fine-tuning pipeline for standard entity schemas (e.g., PERSON, ORG, LOC).
* `Medical_NER.ipynb`: Adapted preprocessing and entity schema for medical texts (e.g., DISEASE, DRUG, SYMPTOM).
* `Legal_NER.ipynb`: Adapted preprocessing and entity schema for legal documents (e.g., COURT, JUDGE, PROVISION).

> **Note:** The major difference across these notebooks is strictly the dataset preprocessing logic and the specific BIO/IOB entity schema tailored to that domain. 

##  Requirements

To run these notebooks, you will need the following dependencies installed:

```bash
pip install torch transformers datasets seqeval scikit-learn
```

##  Usage

1. Clone the repository to your local machine or server.
2. Ensure you have the required datasets downloaded (or link your Hugging Face Datasets/local files in the data loading sections of the notebooks).
3. Open the notebook corresponding to your target domain.
4. Run the cells sequentially. The notebooks will handle data preparation, train the model, and output a detailed classification report at the end.

## Evaluation Metrics

Because standard accuracy is often artificially inflated in NER tasks due to the high frequency of `O` (Outside) tokens, this pipeline relies on `seqeval` to generate entity-level metrics. The evaluation step provides:
* **Precision**: How many of the predicted entities are correct?
* **Recall**: How many of the actual entities did the model find?
* **F1-Score**: The harmonic mean of Precision and Recall.
* **Raw Token Accuracy**: Provided as a baseline metric.
