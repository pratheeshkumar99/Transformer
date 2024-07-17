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

# Neural Machine Translation (NMT) Training

## Training Process and Configuration

This project leverages the OPUS-books dataset for English to Italian translation, provided by Hugging Face, to train a Neural Machine Translation (NMT) model using the Transformer architecture.

### Data Preparation

#### Dataset Requirements
- **Content**: The dataset consists of parallel sentences in English and Italian, suitable for training machine translation systems.
- **Source**: Data is sourced from the OPUS-books collection available through the Hugging Face datasets library.

#### Preprocessing Steps
1. **Tokenization**: Sentences are tokenized into words or subwords, which are crucial for processing by the Transformer model. The tokenization process is handled using the `tokenizers` library from Hugging Face, specifically utilizing `WordLevel` tokenization. The script includes a function (`get_or_build_tokenizer`) that either loads an existing tokenizer or trains a new one from the dataset if it doesn't already exist.

2. **Cleaning**: Ensure data quality by removing noisy or irrelevant characters and correcting common spelling errors. While the current script primarily focuses on tokenization, any additional cleaning operations, such as removing unwanted characters or correcting spelling errors, can be integrated into the preprocessing pipeline before tokenization.

3. **Splitting**: The dataset is typically divided into training, validation, and test sets. A common division might allocate 80% for training, 10% for validation, and 10% for testing. In the script, this is managed using the `random_split` method from PyTorch's `torch.utils.data`. The dataset is split into training and validation sets, and each subset is wrapped into a custom `BilingualDataset` class, which prepares the data for the model.

### Configuration Settings
The training parameters can be configured in the `config.py` file. Key settings include:

- **`batch_size`**: The number of samples per batch of computation. 
    batch_size = 8

- **`num_epochs`**: The number of complete passes through the training dataset. For instance:
  ```python
  num_epochs = 20  # Train the model for 20 epochs.
  learning_rate = 10**-4, #Train the model with this learning rate.

-**`lr`**: Learning rate for the optimizer
    lr = 10**-4

-**`seq_len`**: The length of the input and output sequences.
    seq_len = 350

-**`d_model`**: The dimension of the model.
    d_model = 512

-**`datasource`**: The source of the dataset
    datasource = 'opus_books'

-**`lang_src`**: The source language code.
    lang_src = "en"

-**`lang_tgt`**: The target language code
    lang_tgt = "it"

-**`model_folder`**: The folder to save model checkpoints
    model_folder = "/content/drive/MyDrive/Models/pytorch-transformer/weights"

-**`model_basename`**: The base name for model checkpoints.
    model_basename = "tmodel_"

-**`preload`**: The model checkpoint to preload
    preload = "latest"

-**`tokenizer_file`**: The file pattern for tokenizers.
    tokenizer_file = "tokenizer_{0}.json"experiment_name: The name for the experiment logs

## Training Script Overview

The training script is designed to load the dataset, preprocess the data, configure the model, and handle the training loop. Here are the main components:

- **Data Loading and Tokenization**: The dataset is loaded from Hugging Face’s library and tokenized using the tokenizers library. Tokenizers are built or loaded for both source and target languages.
- **Dataset Preparation**: The dataset is split into training and validation sets. A custom dataset class (BilingualDataset) is used to prepare the data, including adding start-of-sequence and end-of-sequence tokens and padding the sequences to a fixed length.
- **Model Training**: The script sets up the Transformer model, optimizer, and loss function. It handles the training loop, including logging progress and running validation checks at the end of each epoch. The model’s performance is tracked using metrics such as Character Error Rate (CER), Word Error Rate (WER), and BLEU score.

## Running the Training

To run the training process, follow these steps:

1. **Install Dependencies**: Ensure you have all the necessary Python packages installed. This can typically be done via a `requirements.txt` file or using a package manager like `pip`.
2. **Configure Parameters**: Adjust the configuration settings in `config.py` to match your requirements, such as the number of epochs, learning rate, and dataset paths.
3. **Execute the Script**: Run the main training script. This will load the dataset, preprocess the data, train the model, and save the model checkpoints and logs.

## Additional Information

- **Validation and Testing**: During the training process, the model’s performance is periodically evaluated on the validation set. After training, the model can be tested on a separate test set to measure its accuracy and generalization capability.
- **TensorBoard Integration**: The script includes integration with TensorBoard for tracking training progress and visualizing metrics. Ensure TensorBoard is installed and configured to use this feature.
- **Error Handling and Debugging**: The script includes various checks and logging statements to help with debugging. Ensure to review these logs if you encounter any issues during the training process.

By following these steps and understanding the preprocessing and configuration details, you can effectively train a Neural Machine Translation model using the provided scripts and dataset.

## File Descriptions

- **attention_visual.ipynb**: A Jupyter notebook for visualizing the model’s attention mechanisms.
- **Beam_Search.ipynb**: A Jupyter notebook implementing beam search strategies for better translations.
- **Colab_Train.ipynb**: A Jupyter notebook for training the model on Google Colab.
- **config.py**: Configuration file containing the training parameters.
- **dataset.py**: Script for preparing and loading the dataset.
- **Inference.ipynb**: A Jupyter notebook for running inference on the trained model.
- **model_train.ipynb**: A Jupyter notebook for training the model interactively.
- **model.py**: Script defining the Transformer model architecture.
- **README.md**: This file.
- **requirements.txt**: List of required Python packages.
- **runs/**: Directory containing TensorBoard logs.
- **tokenizer_en.json**: Pretrained tokenizer for the source language.
- **tokenizer_it.json**: Pretrained tokenizer for the target language.
- **train_.py**: Main script for training the model.
- **translate.py**: Script for running translation on input text files.