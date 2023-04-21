from share import *
from cldm.hack import hack_everything

hack_everything(clip_skip=2)

import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import argparse

preprocessor = None

model_name = 'control_v11p_sd15_openpose'
any_model_name = 'control_any3_openpose'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(
    load_state_dict('/Users/liuyanjun/workspace/youtube/opencv/ControlNet/models/v1-5-pruned.ckpt', location='cpu'),
    strict=False)
model.load_state_dict(
    load_state_dict(f'/Users/liuyanjun/workspace/youtube/opencv/ControlNet/models/{any_model_name}.pth',
                    location='cpu'), strict=False)
model = model.cpu()
ddim_sampler = DDIMSampler(model)

apply_openpose = OpenposeDetector()


def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
            guess_mode, strength, scale, seed, eta):
    global preprocessor

    if 'Openpose' in det:
        if not isinstance(preprocessor, OpenposeDetector):
            preprocessor = OpenposeDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), hand_and_face='Full' in det)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        shape = (4, H // 8, W // 8)
        return control, shape


def generate_cond(prompt, a_prompt, n_prompt, num_samples):
    cond_c_crossattn = [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
    un_cond_c_crossattn = [model.get_learned_conditioning([n_prompt] * num_samples)]

    return cond_c_crossattn, un_cond_c_crossattn


def generate_images(num_samples, ddim_steps, guess_mode, strength, scale, eta, control, cond_c_crossattn,
                    un_cond_c_crossattn, shape):
    cond = {"c_concat": [control],
            "c_crossattn": cond_c_crossattn}
    un_cond = {"c_concat": None if guess_mode else [control],
               "c_crossattn": un_cond_c_crossattn}
    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
            [strength] * 13)
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                       255).astype(
        np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results


def parse_args():
    """
    :return:
    input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode,
    strength, scale, seed, eta
    """
    desc = "Transform reality video to anime video by using CPU torch <control NET>"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, help='video file')
    parser.add_argument('--image', type=str, help='image file')
    parser.add_argument('--det', type=str, choices=["Openpose_Full", "Openpose", "None"], default="Openpose_Full")
    parser.add_argument('--prompt', type=str, default="Bob hair, by Makoto Shinkai ", help='prompt')
    parser.add_argument('--a_prompt', type=str, default="best quality, extremely detailed", help='prompt')
    parser.add_argument('--n_prompt', type=str, default="longbody, lowres, bad anatomy, bad hands, missing "
                                                        "fingers, extra digit, fewer digits, cropped, worst quality, "
                                                        "low quality", help='prompt')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--image_resolution', type=int, default=512)
    parser.add_argument('--detect_resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--guess_mode', type=bool, default=False)
    parser.add_argument('--strength', type=int, default=1)
    parser.add_argument('--scale', type=int, default=9)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eta', type=int, default=0.0)

    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    if arg.seed:
        seed = arg.seed
    else:
        seed = random.randint(0, 2147483647)

    if arg.image:
        raise NotImplementedError("Choose --video")
    elif arg.video:
        generate_cond_kwargs = {
            "prompt": arg.prompt,
            "a_prompt": arg.a_prompt,
            "n_prompt": arg.n_prompt,
            "num_samples": arg.num_samples
        }
        cond_c_crossattn, un_cond_c_crossattn = generate_cond(**generate_cond_kwargs)
        cap = cv2.VideoCapture(arg.video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Totol frame: {length}")
        count = 0
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            count += 1
            if ret:
                process_kwargs = {
                    "det": arg.det,
                    "input_image": frame,
                    "prompt": arg.prompt,
                    "a_prompt": arg.a_prompt,
                    "n_prompt": arg.n_prompt,
                    "num_samples": arg.num_samples,
                    "image_resolution": arg.image_resolution,
                    "detect_resolution": arg.detect_resolution,
                    "ddim_steps": arg.ddim_steps,
                    "guess_mode": arg.guess_mode,
                    "strength": arg.strength,
                    "scale": arg.scale,
                    "seed": seed,
                    "eta": arg.eta
                }
                control, shape = process(**process_kwargs)
                generate_images_kwargs = {
                    "num_samples": arg.num_samples,
                    "ddim_steps": arg.ddim_steps,
                    "guess_mode": arg.guess_mode,
                    "strength": arg.strength,
                    "scale": arg.scale,
                    "eta": arg.eta,
                    "control": control,
                    "cond_c_crossattn": cond_c_crossattn,
                    "un_cond_c_crossattn": un_cond_c_crossattn,
                    "shape": shape
                }
                results = generate_images(**generate_images_kwargs)
                for result in results:
                    output = Image.fromarray(result)
                    output.save(f"ouput_image/output-{count}.png")
            else:
                break

    else:
        raise NotImplementedError("Choose --images or --video")
