import torch
import torch.nn as nn

from .tokenizer.tokenizer import Tokenizer, Encoder, Decoder
from .tokenizer.nets import EncoderDecoderConfig
from .rnn import LM


class Sim(nn.Module):
    def __init__(self):
        super().__init__()
        config = EncoderDecoderConfig(
            resolution=64,
            in_channels=3,
            z_channels=512,
            ch=64,
            ch_mult=[1,1,1,1,1]   ,
            num_res_blocks=2,
            attn_resolutions=[8,16],
            out_ch=3,
            dropout=0.0
        )

        encoder = Encoder(config)
        decoder = Decoder(config)
        self.tokenizer = Tokenizer(vocab_size=512, embed_dim=512, encoder=encoder, decoder=decoder)

        self.world_model = LM(
            vocab_size=517, # pad + 4 actions + 512 tokens
            emb_dim=1024,
            hidden_dim=1024,
            num_layers=3
        )


def read_iris_tokenizer_state(iris_checkpoint='iris.pt'):
    state_dict = torch.load(iris_checkpoint, map_location='cpu')
    state_dict = {k[len("tokenizer."):]: v for k, v in state_dict.items() if k.startswith('tokenizer.')}
    return state_dict


def load_world_model(world_model_checkpoint='pool4096_1.pt'):
    ckpt = torch.load(world_model_checkpoint, map_location='cpu')
    model = LM(vocab_size=int(ckpt['args']['vocab']),
               emb_dim=ckpt['args']['rnn_size'],
               hidden_dim=ckpt['args']['rnn_size'],
               num_layers=ckpt['args']['num_layers'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


if __name__ == '__main__':
    sim = Sim()
    sim.tokenizer.load_state_dict(read_iris_tokenizer_state(), strict=False)
    sim.world_model.load_state_dict(load_world_model().state_dict())
    torch.save(sim.state_dict(), 'twist-rollout.pt')