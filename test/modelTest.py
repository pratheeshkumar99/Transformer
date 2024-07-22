
import sys
from pathlib import Path  
import torch  

try:
    # Add the path to the sys.path
    current_directory = Path(__file__).parent

    # Get the parent directory
    parent_directory = current_directory.parent

    # Add the parent directory to sys.path
    sys.path.append(str(parent_directory))

    import model,dataset
    from model import build_transformer
    from dataset import causal_mask
except Exception as e:
    print(f"An error occurred: {e}")


def create_model():
    try:
        model = build_transformer(src_vocab_size=1000, tgt_vocab_size=1000, d_model=512, num_layers=6, h=8, dropout=0.1, d_ff=2048, src_seq_len=200, tgt_seq_len=250)
        assert model is not None
        print("Model created successfully!")
        return model
    except Exception as e:
        print(f"An error occurred: {e}")


def test_encoder(model, example_src, example_src_mask):
    try:
        assert example_src_mask.shape == (batch_size,1, src_seq_len)
        assert example_src.shape == (batch_size, src_seq_len)
        encoder_output = model.encode(example_src, example_src_mask)
        assert encoder_output.shape == (batch_size, src_seq_len, d_model)
        print("Encoder works correctly!")
        return encoder_output
    except AssertionError as e:
            print(f"Assertion Error: {e}")
    except Exception as e:
            print(f"An error occurred: {e}")


def test_decoder(model,encoder_output,example_tgt,example_src_mask,example_tgt_mask):
    try:
        assert example_tgt.shape == (batch_size, trg_seq_len)
        assert example_src_mask.shape == (batch_size, 1, src_seq_len)
        assert example_tgt_mask.shape == (batch_size, trg_seq_len, trg_seq_len)
        decoder_output = model.decode(encoder_output,example_src_mask, example_tgt,example_tgt_mask)
        assert decoder_output.shape == (batch_size, trg_seq_len, d_model)
        print("Decoder works correctly!")
        return decoder_output

    except AssertionError as e:
        print(f"Assertion Error: {e}")

def project_test(model,decoder_output):
    try:
        output = model.project(decoder_output)
        assert output.shape == (batch_size, trg_seq_len, target_vocab_size)
        print("Project works correctly!")
        return output
    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_caual_mask(seq_len):
    try:
        mask = causal_mask(seq_len)
        assert mask.shape == (1, seq_len, seq_len)
        print("Causal mask works correctly!")
        return mask
    except AssertionError as e:
        print(f"Assertion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    batch_size = 64
    src_seq_len = 200
    trg_seq_len = 250
    d_model = 512
    target_vocab_size = 1000
    example_src = torch.randint(low=0, high=1000, size=(batch_size, src_seq_len))
    example_tgt = torch.randint(low=0, high=1000, size=(batch_size, trg_seq_len))
    example_src_mask = torch.ones(batch_size, 1,src_seq_len)
    
    example_tgt_pad_mask = torch.randint(0,100,(batch_size,trg_seq_len))
    example_tgt_pad_mask = example_tgt_pad_mask.unsqueeze(1) # (batch_size, 1, trg_seq_len)
    example_tgt_causal_mask = create_caual_mask(trg_seq_len) # (1, trg_seq_len, trg_seq_len)
    example_tgt_mask = example_tgt_pad_mask & example_tgt_causal_mask #(batch_size, trg_seq_len, trg_seq_len)
    
    
    model = create_model()
    encoder_output = test_encoder( model,example_src, example_src_mask) # (batch_size, src_seq_len, d_model)
    decoder_output = test_decoder(model,encoder_output, example_tgt, example_src_mask, example_tgt_mask) # (batch_size, trg_seq_len, d_model)
    project_test(model,decoder_output) # (batch_size, trg_seq_len, target_vocab_size)


