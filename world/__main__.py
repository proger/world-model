import cv2
import torch

from .sim import Sim


class InteractiveSim:
    def __init__(self, sim):
        self.tokenizer = sim.tokenizer
        self.world_model = sim.world_model
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

    def put_help(self, image):
        image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_position = (10, image.shape[0] - 10)  # 10 pixels from the left and bottom
        text = 'arrow keys to move around; space to reset; r to truncate history; q to exit'
        cv2.putText(image, text, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        return image

    def append_text(self, image, text):
        image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
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
                image = self.put_help(image)
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
                    image = self.render(current_frame_tokens)
                    image = self.postprocess_image(image)
                    image = self.append_text(image, '!')
                else:
                    self.state = state
                    current_frame_tokens = new_frame_tokens
                    image = self.postprocess_image(image)
                    image = self.append_text(image, self.action_to_tag[action])

            cv2.imshow('sim', image)
            key = cv2.waitKey(100) & 0xff

        cv2.destroyAllWindows()


def main():
    sim = Sim()
    sim.load_state_dict(torch.hub.load_state_dict_from_url('https://huggingface.co/darkproger/twist-rollout/resolve/main/twist-rollout.pt'))
    interactive = InteractiveSim(sim)
    interactive.loop()


if __name__ == '__main__':
    main()
