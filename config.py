from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "/content/drive/MyDrive/Models/pytorch-transformer/weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}_{config['datasource']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_weights_file_name(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(model_folder,config):
    model_folder = model_folder
    model_filename = f"{config['model_basename']}*"
    print(model_folder,model_filename)
    weights_files = list(Path(model_folder).glob(model_filename))
    print(weights_files)
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
