# Transformer Model for Neural Machine Translation

## Description
This project implements a Transformer model designed for translating text from English to Kannada, based on the architecture proposed in the "Attention is All You Need" paper. The implementation utilizes PyTorch, along with Hugging Face's `datasets` and `tokenizers` libraries, to create a robust and efficient translation system suitable for the specific linguistic features of the Kannada language.

## Quick Start

### Clone the Repository
Clone the repository to get started with the Transformer model:
```bash
git clone https://github.com/pratheeshkumar99/Transformer.git
cd Transformer

```

### Install Dependencies
Install the necessary Python packages:
```bash
pip install -r requirements.txt
```
### Run a Quick Translation

### Translate text from a file using:

```bash
python translate.py --input your_input.txt --output output.txt
```

## Additional Scripts

- **Visualization**: Explore the model's attention mechanisms interactively by navigating to [attention_visual.ipynb](attention_visual.ipynb). This notebook provides tools to visually analyze how different parts of the input sequence affect each other through the model's attention weights.
  
- **Beam Search**: Enhance the decoding process with [Beam_Search.ipynb](Beam_Search.ipynb), which implements beam search strategies for more accurate and coherent translations. This technique is particularly useful for complex translation tasks and when you need higher quality outputs.


# Training Process and Configuration

## Data Preparation
The training leverages the OPUS-books dataset for English to Italian translation, provided by Hugging Face, which is instrumental for training our Neural Machine Translation (NMT) model.

### Dataset Requirements
- **Content**: The dataset consists of parallel sentences in English and Italian, suitable for training machine translation systems.
- **Source**: Data is sourced from the OPUS-books collection available through the Hugging Face datasets library.

