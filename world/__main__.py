import cv2

import torch

from .tokenizer.tokenizer import Tokenizer, Encoder, Decoder
from .tokenizer.nets import EncoderDecoderConfig
from .rnn import LM


def load_image_tokenizer(iris_checkpoint='iris.pt'):
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
    tokenizer = Tokenizer(vocab_size=512, embed_dim=512, encoder=encoder, decoder=decoder)

    state_dict = torch.load(iris_checkpoint, map_location='cpu')
    state_dict = {k[len("tokenizer."):]: v for k, v in state_dict.items() if k.startswith('tokenizer.')}
    tokenizer.load_state_dict(state_dict, strict=False)
    tokenizer.eval()
    return tokenizer


def load_world_model(world_model_checkpoint='pool4096_1.pt'):
#def load_world_model(world_model_checkpoint='megapool_1.pt'):
    ckpt = torch.load(world_model_checkpoint, map_location='cpu')
    model = LM(vocab_size=int(ckpt['args']['vocab']),
               emb_dim=ckpt['args']['rnn_size'],
               hidden_dim=ckpt['args']['rnn_size'],
               num_layers=ckpt['args']['num_layers'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


class Sim:
    def __init__(self):
        self.tokenizer = load_image_tokenizer()
        self.world_model = load_world_model()
        self.frame_offset = 5
        self.key_to_action = {
            0: 2, # forward
            1: 1, # pad/stop
            2: 4, # left
            3: 3, # right
        }
        self.action_to_tag = {
            2: '^',
            1: '.',
            4: '<',
            3: '>',
        }

        self.flower = torch.LongTensor([0, 452, 297,  50, 438, 112, 331, 2, 430,  39, 471,  58, 476,  59, 501, 358, 331])
        self.tag_history = ""

    def reset(self, initial_frame=None):
        if initial_frame is None:
            initial_frame = self.flower
        logits, self.state = self.world_model(
            initial_frame[:, None] + self.frame_offset,
            self.world_model.init_hidden()
        )
        print('reset to', initial_frame)
        best_actions = torch.topk(logits[[-1],:], 4)
        print('best actions after reset', best_actions)
        self.initial_frame = initial_frame[-16:]
        return self.initial_frame

    def render(self, frame_tokens):
        #frame = np.random.RandomState(action).randint(0, 256, (64, 64, 3), dtype=np.uint8)

        b, e, h, w = 1, 512, 4, 4
        z_q = self.tokenizer.embedding(frame_tokens).reshape(b, h, w, e).permute(0, 3, 1, 2).contiguous()
        frame = self.tokenizer.decode(z_q, should_postprocess=True).squeeze().permute(1, 2, 0).numpy()
        return frame

    def step(self, state, action):
        x = torch.LongTensor([action])[:, None]

        # autoregressively generate one image frame
        for k in range(16):
            logits, state = self.world_model(x[[-1], :], state)
            v, _ = torch.topk(logits, 1)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            ix = probs.multinomial(num_samples=1)
            x = torch.cat([x, ix], dim=0)

        logits, _state = self.world_model(x[[-1], :], state)
        best_actions = torch.topk(logits, 4)

        print(x.squeeze(), 'best_action', best_actions)
        return x[-16:, 0] - self.frame_offset, state

    def postprocess_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
        return frame

    def put_text(self, image, text):
        image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        text_position = (10, image.shape[0] - 10)  # 10 pixels from the left and bottom
    
        self.tag_history += text
        self.tag_history = self.tag_history[-24:]
        text = self.tag_history

        cv2.putText(image, text, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        return image

    @torch.inference_mode()
    def loop(self):
        key = 32

        while True:
            if key != 0xff:
                print('key', key)

            if key in [27, 113]:  # 'ESC' or 'q' to exit
                break
            elif key == 32: # space to reset
                current_frame_tokens = self.reset()
                image = self.render(current_frame_tokens)
                image = self.postprocess_image(image)
            elif key == 114: # r to reset hidden state from current_frame_tokens
                current_frame_tokens = self.reset(initial_frame=current_frame_tokens)
                image = self.render(current_frame_tokens)
                image = self.postprocess_image(image)
                print('reset hidden state', current_frame_tokens)
            elif key in [0, 1, 2, 3]:
                action = self.key_to_action[key]

                new_frame_tokens, state = self.step(self.state, action)
                try:
                    image = self.render(new_frame_tokens)
                except IndexError:
                    print('index error, ignoring action')
                else:
                    self.state = state
                    current_frame_tokens = new_frame_tokens
                    image = self.postprocess_image(image)
                    image = self.put_text(image, self.action_to_tag[action])

            cv2.imshow('sim', image)
            key = cv2.waitKey(100) & 0xff

        cv2.destroyAllWindows()


if __name__ == '__main__':
    sim = Sim()
    sim.loop()

