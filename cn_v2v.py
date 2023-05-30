
"""
ControlNet only video2video webui custom script

base code adapted from https://github.com/Filarius/video2video

"""

import math

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat

import os
import sys
from subprocess import Popen, PIPE
import numpy as np
from PIL import Image
from random import randint
import platform

import gradio as gr

import ldm

import modules
import modules.images as images

from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules import script_callbacks
from modules import scripts, shared
from modules.processing import Processed, process_images, StableDiffusionProcessingImg2Img
from modules import processing
from modules.shared import state, opts


try:
    import skvideo
except:
    if 'VENV_DIR' in os.environ:
        path = os.path.join(os.environ["VENV_DIR"],'scripts','python')
    elif 'PYTHON' in os.environ:
        path = os.path.join(os.environ["PYTHON"], 'python')
    else:
        from shutil import which
        path = which("python")
    os.system(path+ " -m pip install sk-video")
    import skvideo


class CrossFrameAttnLatentMemory:
    def __init__(self):
        self.init()

        self.max_cache_frame = 2

        self.enabled = False
        self.attn_first_frame = True

        self.attn_2_frames = True
        self.attn_3_frames = False

        self.shot_frame_pos = [0]
        self.frame_denoising_strength = {}
        self.full_or_blur_shot_denoising_strength = 0.1 

    def init(self):
        self.cur_frame = -1
        self.last_sampling_step = -1
        self.cur_layer = -1

        self.latents_key = []
        self.latents_value = []

    def set_shot_frame_pos(self, val):
        val = val.strip().replace('，', ',')
        for v in val.split(','):
            v = v.strip()
            if v == '':
                continue

            if v[-1] == 'F':
                pos = int(v[:-1])
                self.frame_denoising_strength[pos] = self.full_or_blur_shot_denoising_strength
            else:
                pos = int(v)

            self.shot_frame_pos.append(pos)

        assert len(self.shot_frame_pos) > 0

    def get_denoising_strength(self, orig_denoising_strength):
        reset_first_store = self.can_reset_first_store()
        if reset_first_store:
            return self.frame_denoising_strength.get(self.cur_frame, orig_denoising_strength)

        shot_frame = self.frame_in_shot(self.cur_frame)
        if shot_frame == -1:
            print('================== get_denoising_strength shot_frame %d =================' % shot_frame)
        return self.frame_denoising_strength.get(shot_frame, orig_denoising_strength)

    def can_store(self):
        reset_last_store = False
        reset_first_store = False
        if not self.attn_first_frame:
            return True, reset_last_store, reset_first_store

        if self.cur_frame == 0:
            return True, reset_last_store, reset_first_store

        if self.attn_2_frames or self.attn_3_frames:
            reset_last_store = self.cur_frame > 1
            return True, reset_last_store, self.can_reset_first_store()

        return False, reset_last_store, reset_first_store

    def can_reset_first_store(self, frame=None):
        frame = self.cur_frame if frame is None else frame
        return frame in self.shot_frame_pos

    def frame_in_shot(self, frame):
        l = len(self.shot_frame_pos)
        for i, v in enumerate(self.shot_frame_pos):
            if i == l - 1 or frame >= v and frame < self.shot_frame_pos[i + 1]:
                return v

        return -1

    def set_frame(self):
        self.cur_frame += 1
        self.last_sampling_step = -1
        self.cur_layer = -1

    def set_step(self, sampling_step):
        can_store, reset_last_store, _ = self.can_store()
        if can_store and not reset_last_store:
            # sampling_step update after model.forwad, so lagged behind, we must deal it self 
            # if sampling_step == 0:
            if self.last_sampling_step == -1:
                self.latents_key.append([])
                self.latents_value.append([])

            self.latents_key[-1].append([])
            self.latents_value[-1].append([])

        # self.last_sampling_step = sampling_step
        self.last_sampling_step += 1
        self.cur_layer = -1

    def set_layer(self):
        self.cur_layer += 1

    def put_key(self, latent):
        can_store, reset_last_store, reset_first_store = self.can_store()
        if reset_first_store:
            self.latents_key[0][self.last_sampling_step][self.cur_layer] = latent
        elif reset_last_store:
            if self.attn_3_frames:
                # XXX: put must after get
                self.latents_key[-1][self.last_sampling_step][self.cur_layer] = latent
        elif can_store:
            self.latents_key[-1][-1].append(latent)

    def put_value(self, latent):
        can_store, reset_last_store, reset_first_store = self.can_store()
        if reset_first_store:
            self.latents_key[0][self.last_sampling_step][self.cur_layer] = latent
        elif reset_last_store:
            if self.attn_3_frames:
                # XXX: put must after get
                self.latents_value[-1][self.last_sampling_step][self.cur_layer] = latent
        elif can_store:
            self.latents_value[-1][-1].append(latent)

    def get_key(self, default):
        return self.get(default, self.latents_key)

    def get_value(self, default):
        return self.get(default, self.latents_value)

    # default is cur frame
    def get(self, default, latents):
        if self.can_reset_first_store():
            return default

        if self.attn_first_frame:
            if self.cur_frame == 0:
                return default

            # first frame
            h = latents[0][self.last_sampling_step][self.cur_layer]

            if self.attn_2_frames:
                return torch.cat((default, h), dim=1)

            if self.attn_3_frames:
                return self._attn_3_frames(default, h, latents)

            return h
        else:
            if len(latents) < 2:
                return default

            return latents[-2][self.last_sampling_step].pop(0)

    def _attn_3_frames(self, default, first_h, latents):
        assert self.cur_frame > 0

        if self.cur_frame == 1 or self.can_reset_first_store(self.cur_frame - 1):
            return torch.cat((default, first_h), dim=1)

        prev_h = latents[-1][self.last_sampling_step][self.cur_layer]
        return torch.cat((prev_h, default, first_h), dim=1)


class Script(scripts.Script):
    save_dir = "outputs/img2img-video-cn_v2v/"

    def title(self):
        return 'CN v2v'

    def show(self, is_img2img):
        return is_img2img

    def __init__(self):
        self.img2img_component = gr.Image()
        self.img2img_inpaint_component = gr.Image()
        self.is_have_callback = False

    def ui(self, is_visible):
            def img_dummy_update(arg):
                if arg is None:
                    import io
                    import base64

                    dummy_image = io.BytesIO(base64.b64decode(DUMMY_IMAGE))
                    img = Image.open(dummy_image)
                    return img
                else:
                    return arg

            with gr.Row():
                file_obj = gr.File(label="Upload Video", file_types = ['.*;'], live=True, file_count = "single")

            with gr.Row():
                with gr.Column(min_width=100):
                    use_optimized_preset = gr.Checkbox(
                            label='Use optimized preset, all other settings will be override, except temporalnet/prompt/width/height/seed/model. (Have to Set `Multi ControlNet: Max models amount` to 6)',
                            value=True)

                  # XXX simply set temporalnet_weight to 0.2 for more cartoon
                  # use_more_cartoon_preset = gr.Checkbox(
                  #         label='Add more cartoon, all other settings include temporalnet will be override, except prompt/width/height/seed/model. (Set `Multi ControlNet: Max models amount` to 10, you can change it in browser debug mode)',
                  #         value=False)

                    test_single_image = gr.Checkbox(
                            label='Test settings on a single image, have to upload a image in img2img',
                            value=False)

                    use_temporalnet = gr.Checkbox(label='Use temporalnet', value=True)
                    temporalnet_weight = gr.Slider(
                        label="Temporalnet weight",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.5,
                    )
                    temporalnet_control_mode = gr.Radio(
                            choices=['Balanced', 'My prompt is more important', 'ControlNet is more important'], 
                            value='Balanced',
                            label="Temporalnet Control Mode (Guess Mode)")

                    # good for grayscale video
                    # Taylor Swift and Tom Hanks are talking before a white wall, By Van Gogh 
                    # best checkpoint are Protogen_v2.2 cardosanimeV2.hCCt
                    use_cross_frame_attn = gr.Checkbox(label='Use cross frame attn', value=False)

            with gr.Row():
                fps = gr.Slider(
                    label="FPS",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=24,
                )

            file_obj.upload(fn=img_dummy_update, inputs=[self.img2img_component], outputs=[self.img2img_component])

            refresh_link = gr.Button('Refresh link (Must click before generate)')
            img_link = gr.HTML(value='', visible=True)
            video_link = gr.HTML(value='', visible=True)

            file_obj.change(fn=self.build_img_video_link, inputs=[file_obj], 
                    outputs=[video_link, img_link])
            refresh_link.click(fn=self.build_img_video_link, inputs=[file_obj], 
                    outputs=[video_link, img_link])

            return [
                    fps,
                    file_obj,
                    use_temporalnet,
                    temporalnet_weight, 
                    temporalnet_control_mode,
                    use_optimized_preset,
                    # use_more_cartoon_preset,
                    test_single_image,
                    use_cross_frame_attn,
                    ]

    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_image":
            self.img2img_component = component
            return self.img2img_component

    def run(self, 
            p:StableDiffusionProcessingImg2Img,
            fps,
            file_obj,
            use_temporalnet,
            temporalnet_weight,
            temporalnet_control_mode,
            use_optimized_preset,
            # use_more_cartoon_preset,
            test_single_image,
            use_cross_frame_attn,
            *args):

            path = modules.paths.script_path
            if platform.system() == 'Windows':
                ffmpeg.install(path)
                import skvideo
                skvideo.setFFmpegPath(os.path.join(path, "ffmpeg"))
            import skvideo.io

            processor_res = 512
            use_more_cartoon_preset = False

            if use_more_cartoon_preset:
                use_temporalnet = True

            if test_single_image:
                return run_img2img(
                        p,
                        use_optimized_preset=use_optimized_preset, 
                        use_temporalnet=use_temporalnet,
                        processor_res=processor_res,
                        temporalnet_control_mode=temporalnet_control_mode,
                        temporalnet_weight=temporalnet_weight,
                        use_more_cartoon_preset=use_more_cartoon_preset,
                        use_cross_frame_attn=use_cross_frame_attn,
                        )

            LATENT_MEM.enabled = use_cross_frame_attn

            if use_cross_frame_attn:
                LATENT_MEM.init()
                LATENT_MEM.attn_first_frame = True
                LATENT_MEM.attn_2_frames = True

            if not self.is_have_callback:
                def cross_frame_attn_callback(params): # CFGDenoiserParams
                    LATENT_MEM.set_step(params.sampling_step)

                if use_cross_frame_attn:
                    on_cfg_denoiser(cross_frame_attn_callback)

                self.is_have_callback = True

            input_file = os.path.normpath(file_obj.name.strip())

            output_file = os.path.basename(input_file)
            output_file = os.path.splitext(output_file)[0]

            output_basename = get_output_basename(output_file, self.save_dir)
            output_file = f'{self.save_dir}/{output_basename}.mp4'
            print(output_file)

            decoder = get_video_decoder(input_file, fps)
            encoder = get_video_encoder(output_file, fps)

            initial_seed = p.seed
            if initial_seed == -1:
                initial_seed = randint(100000000,999999999)
                p.seed = initial_seed
            processing.fix_seed(p)

            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.batch_count = 1

            state.job_count = decoder.inputframenum
            job_i = 0
            state.job_no = job_i

            last_gen_image = None

            if use_optimized_preset:
                preset_img2img_for_colorful(p, use_more_cartoon_preset, use_cross_frame_attn)
                make_control_nets_for_colorful(p, processor_res, use_more_cartoon_preset, use_cross_frame_attn)

                if use_temporalnet:
                    make_control_net_temporalnet_for_colorful(
                            p,
                            last_gen_image=last_gen_image,
                            temporalnet_weight=temporalnet_weight,
                            temporalnet_control_mode=temporalnet_control_mode, 
                            processor_res=processor_res,
                            use_more_cartoon_preset=use_more_cartoon_preset,
                            use_cross_frame_attn=use_cross_frame_attn)

            batch = []
            is_last = False
            frame_generator = decoder.nextFrame()

            while not is_last:

                LATENT_MEM.set_frame()

                try:
                    raw_image = next(frame_generator,[])
                except:
                    print('================ next frame_generator failed ==================')
                    raw_image = []

                image_PIL = None
                if len(raw_image)==0:
                    is_last = True
                else:
                    image_PIL = Image.fromarray(raw_image,mode='RGB')
                    batch.append(image_PIL)

                if (len(batch) == p.batch_size) or ( (len(batch) > 0) and is_last ):
                    p.seed = initial_seed
                    p.init_images = batch

                    batch = []
                    try:
                        proc = process_images(p)

                        # first gen, for save params in meta
                        if last_gen_image is None:
                            save_image(p, proc, self.save_dir, output_basename)
                    except Exception as e:
                        import traceback
                        print('================ process_images failed ==================')
                        traceback.print_exc()
                        break

                    # images[1:] is controlnet preprocessed images
                    for output in proc.images[:1]:
                        if output.mode != "RGB":
                            output = output.convert("RGB")

                        last_gen_image = np.asarray(output)

                        encoder.writeFrame(last_gen_image.copy())
                        job_i += 1
                        state.job_no = job_i

            encoder.close()
            remove_current_script_callbacks()

            LATENT_MEM.enabled = False

            return Processed(p, [], p.seed, proc.info)

    def build_img_video_link(self, file_obj, is_abs=True):
        if file_obj is None:
            return '', ''

        print('build_img_video_link')
        print(file_obj.name)
        os.makedirs(self.save_dir, exist_ok=True)

        input_file = os.path.normpath(file_obj.name.strip())
        output_file = os.path.basename(input_file)
        output_file = os.path.splitext(output_file)[0]
        output_basename = get_output_basename(output_file, self.save_dir)
        output_video = f'{self.save_dir}/{output_basename}.mp4'
        output_img = f'{self.save_dir}/{output_basename}.png'
        if is_abs:
            # scripts path add /../ is root path
            abs_path = os.path.split(__file__)[0]  + '/../'
            output_video = '/file=' + abs_path + output_video
            output_img = '/file=' + abs_path + output_img

        output_video = f'<a href="{output_video}" target="_blank">{output_video}</a>'
        output_img = f'<a href="{output_img}" target="_blank">{output_img}</a>'

        return output_video, output_img 


def preset_img2img_for_colorful(p, use_more_cartoon_preset, use_cross_frame_attn):
    if p.negative_prompt == '':
        p.negative_prompt = 'deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature'
    p.sampler_name = 'Euler'
    p.denoising_strength = 0.8
    p.cfg_scale = 1
    p.steps = 6

    if use_more_cartoon_preset:
        # p.cfg_scale = 2
        # p.steps = 10
        # p.denoising_strength = 0.5
        p.denoising_strength = 0.8

    if use_cross_frame_attn:
        p.denoising_strength = 1
        p.cfg_scale = 4
        p.steps = 10

    p.resize_mode = 1


def make_control_nets_for_colorful(p, processor_res, use_more_cartoon_preset, use_cross_frame_attn):
    if use_cross_frame_attn:
        return make_control_nets_for_cross_frame_attn(p, processor_res, use_more_cartoon_preset)

    net = p.script_args[1]
    set_control_net_common_settings(net)

    net.enabled = True
    net.control_mode = 'ControlNet is more important'
    net.model = "control_v11p_sd15_lineart_fp16 [5c23b17d]"
    net.module = "lineart_realistic"
    net.weight = 1
    net.processor_res = processor_res
    net.threshold_a = 64
    net.threshold_b = 64


    net = p.script_args[2]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11p_sd15_softedge_fp16 [f616a34f]"
    net.module = "softedge_hedsafe"
    net.weight = 1
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 1
    net.threshold_b = 64


    net = p.script_args[3]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11f1p_sd15_depth_fp16 [4b72d323]"
    net.module = "depth_midas"
    net.weight = 1
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 64
    net.threshold_b = 64


    net = p.script_args[4]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11p_sd15_normalbae_fp16 [592a19d8]"
    net.module = "normal_bae"
    net.weight = 1
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 64
    net.threshold_b = 64


    net = p.script_args[6]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11f1e_sd15_tile_fp16 [3b860298]"
    net.module = "tile_resample"
    net.weight = 1
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 1
    net.threshold_b = 64


    if use_more_cartoon_preset:
        net = p.script_args[7]
        set_control_net_common_settings(net)

        net.enabled = True
        net.model = "control_v11p_sd15_lineart_fp16 [5c23b17d]"
        net.module = "lineart_anime"
        # net.weight = 1
        net.weight = 0.5
        net.guidance_end = 0.5
        net.control_mode = 'ControlNet is more important'
        net.processor_res = processor_res
        net.threshold_a = 64
        net.threshold_b = 64


        net = p.script_args[8]
        set_control_net_common_settings(net)

        net.enabled = True
        net.model = "control_v11p_sd15_canny_fp16 [b18e0966]"
        net.module = "canny"
        # net.weight = 1
        net.weight = 0.5
        net.guidance_end = 0.5
        net.control_mode = 'ControlNet is more important'
        net.processor_res = processor_res
        net.threshold_a = 100
        net.threshold_b = 200


        net = p.script_args[9]
        set_control_net_common_settings(net)

        net.enabled = True
        net.model = "control_v11p_sd15_scribble_fp16 [4e6af23e]"
        net.module = "scribble_hed"
        # net.weight = 1
        net.weight = 0.5
        net.guidance_end = 0.5
        net.control_mode = 'ControlNet is more important'
        net.processor_res = processor_res
        net.threshold_a = 64
        net.threshold_b = 64


        net = p.script_args[10]
        set_control_net_common_settings(net)

        net.enabled = True
        net.model = "control_v11p_sd15_lineart_fp16 [5c23b17d]"
        net.module = "lineart_coarse"
        # net.weight = 1
        net.weight = 0.5
        net.guidance_end = 0.5
        net.control_mode = 'ControlNet is more important'
        net.processor_res = processor_res
        net.threshold_a = 64
        net.threshold_b = 64


def make_control_net_temporalnet_for_colorful(
        p, 
        *,
        last_gen_image=None,
        temporalnet_weight=0.5,
        temporalnet_control_mode='Balanced',
        processor_res=512,
        use_more_cartoon_preset=False,
        use_cross_frame_attn=False):

    net = p.script_args[5]
    set_control_net_common_settings(net)

    net.enabled = True
    # net.image = last_gen_image
    net.loopback = True
    net.model = "diff_control_sd15_temporalnet_fp16 [adc6bd97]"
    net.module = "none"
    net.weight = temporalnet_weight
    net.control_mode = temporalnet_control_mode

    if use_more_cartoon_preset:
        net.weight = 0.2

    net.processor_res = processor_res
    net.threshold_a = 1
    net.threshold_b = 64


def make_control_nets_for_cross_frame_attn(p, processor_res, use_more_cartoon_preset):
    net = p.script_args[1]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11f1p_sd15_depth_fp16 [4b72d323]"
    net.module = "depth_midas"
    net.weight = 0.5
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 64
    net.threshold_b = 64


    net = p.script_args[2]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11p_sd15_lineart_fp16 [5c23b17d]"
    net.module = "lineart_realistic"
    net.weight = 0.5
    net.control_mode = 'Balanced'
    net.processor_res = processor_res
    net.threshold_a = 64
    net.threshold_b = 64


    net = p.script_args[3]
    set_control_net_common_settings(net)

    net.enabled = True
    net.model = "control_v11p_sd15_softedge_fp16 [f616a34f]"
    net.module = "softedge_hed"
    net.weight = 0.5
    net.control_mode = 'ControlNet is more important'
    net.processor_res = processor_res
    net.threshold_a = 1
    net.threshold_b = 64


def set_control_net_common_settings(net):
    params = {'image': None, 'low_vram': False, 'guidance': 1, 'guidance_start': 0, 'guidance_end': 1, 'guess_mode': False, 'pixel_perfect': True, 'resize_mode': 'Crop and Resize', 'is_ui': True, 'batch_images': '', 'output_dir': '', 'loopback': False}

    for k, v in params.items():
        setattr(net, k, v)


def run_img2img(
        p, 
        *,
        use_optimized_preset,
        use_temporalnet,
        temporalnet_weight, 
        temporalnet_control_mode,
        processor_res,
        use_more_cartoon_preset,
        use_cross_frame_attn):
    print('====================== For single img2img ====================')
    if use_optimized_preset:
        preset_img2img_for_colorful(p, use_more_cartoon_preset, use_cross_frame_attn)
        make_control_nets_for_colorful(p, processor_res, use_more_cartoon_preset, use_cross_frame_attn)

        if use_temporalnet:
            make_control_net_temporalnet_for_colorful(
                    p,
                    temporalnet_weight=temporalnet_weight,
                    temporalnet_control_mode=temporalnet_control_mode, 
                    processor_res=processor_res,
                    use_more_cartoon_preset=use_more_cartoon_preset,
                    use_cross_frame_attn=use_cross_frame_attn)

    try:
        proc = process_images(p)
    except Exception as e:
        import traceback
        print('================ process_images failed ==================')
        traceback.print_exc()

    return Processed(p, proc.images, p.seed, proc.info)


def get_video_decoder(input_file, fps):
    decoder = skvideo.io.FFmpegReader(input_file, outputdict={
        '-r':str(fps)
    })

    return decoder


def get_video_encoder(output_file, fps):
    encoder = skvideo.io.FFmpegWriter(
        output_file,
        inputdict={
            '-r': str(fps),
        },
        outputdict={
        '-vcodec': 'libx264',
        # defaut is 1665 kbps
        '-b': '6000000',
        # avoid browser and some player can't play
        '-pix_fmt': 'yuv420p',
    })

    return encoder


def get_output_basename(output_file, save_dir):
    i=1
    while os.path.isfile(f'{save_dir}/{i}_{output_file}.mp4'):
        i+=1
    output_basename = f'{i}_{output_file}'
    return output_basename


def save_image(p, proc, save_dir, basename):
    infotext = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, {}, 0, 0)
    images.save_image(
            proc.images[0],
            save_dir,
            "", 
            p.seed,
            p.all_prompts[0], 
            opts.samples_format,
            info=infotext, 
            p=p,
            forced_filename=basename,
            save_to_dirs=False)


class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        try:
            print(self._cmdln)
            self._process = Popen(
                self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
            )
        except Exception as e:
            print(e)

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def install(path):
        from basicsr.utils.download_util import load_file_from_url
        from zipfile import ZipFile

        ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip"
        ffmpeg_dir = os.path.join(path, "ffmpeg")

        if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, "ffmpeg.exe"))):
            print("Downloading FFmpeg")
            ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)
            with ZipFile(ckpt_path, "r") as zipObj:
                listOfFileNames = zipObj.namelist()
                for fileName in listOfFileNames:
                    if "/bin/" in fileName:
                        zipObj.extract(fileName, ffmpeg_dir)
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffmpeg.exe"),
                os.path.join(ffmpeg_dir, "ffmpeg.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffplay.exe"),
                os.path.join(ffmpeg_dir, "ffplay.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffprobe.exe"),
                os.path.join(ffmpeg_dir, "ffprobe.exe"),
            )

            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin"))
            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
            print("Downloading FFmpeg: Done")
        return

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds


def unload():
    ldm.modules.attention.CrossAttention.forward = ldm.modules.attention.CrossAttention.forward_before_cross_frame
    processing.decode_first_stage = processing.orig_decode_first_stage

if not hasattr(ldm.modules.attention.CrossAttention, 'forward_before_cross_frame'):
   ldm.modules.attention.CrossAttention.forward_before_cross_frame = ldm.modules.attention.CrossAttention.forward


def _cross_frame_forward(self, x, context=None, mask=None):
    """
    Method merge from:
        Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators
        https://arxiv.org/abs/2303.13439

        https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion
    """
    is_cross_attention = context is not None

    if is_cross_attention:
        return ldm.modules.attention.CrossAttention.forward_before_cross_frame(self, x, context, mask)


    h = self.heads

    q = self.to_q(x)
    context = context if is_cross_attention else x
    k = self.to_k(context)
    v = self.to_v(context)

    control_net_model_name = getattr(shared, 'control_net_model_name', '')
    # print('control_net_model_name', control_net_model_name, is_cross_attention)

    # XXX: to avoid cross_frame_attn for temporalnet controlnet,
    #     after update controlnet extension, must add below codes
    #     But before add temporalnet controlnet, no need this, and controlnet also have self_attn 
    # XXX: after disable cross_frame_attn for all controlnet, some frames have multiple hands problem,
    #      but whole quality is a bit better

    # in controlnet.py add below code: forward_param = ControlParams()
    # forward_param.model_name = unit.model

    # in hook.py add below comment: # handle unet injection stuff
    # shared.control_net_model_name = param.model_name
    # control = param.control_model(x=x_in, hint=param.used_hint_cond ...
    # shared.control_net_model_name = ''
    # if not is_cross_attention and LATENT_MEM.enabled and 'temporalnet' not in control_net_model_name:
    # if not is_cross_attention and LATENT_MEM.enabled and control_net_model_name == '':
    if not is_cross_attention and LATENT_MEM.enabled:
        LATENT_MEM.set_layer()

        # XXX: above 15 is another unet forward for negative_prompt
        #      see webui modules/sd_samplers_kdiffusion.py: x_out[-uncond.shape[0]:] = self.inner_model(...
        # if LATENT_MEM.cur_layer < 16 + LATENT_MEM.control_nets_self_attn_layers:
        # if LATENT_MEM.cur_layer < 16:
        if True:

            orig_k = k
            orig_v = v

            k = LATENT_MEM.get_key(orig_k)
            v = LATENT_MEM.get_value(orig_v)

            LATENT_MEM.put_key(orig_k)
            LATENT_MEM.put_value(orig_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

def _replace_forwad(m):
    """
    XXX: This may confick with webui, and disabled optimization, see sd_hijack.py
    """
    if True:
        print('=================== _replace_forwad ===================')
        ldm.modules.attention.CrossAttention.forward = _cross_frame_forward
    else:
        print('=================== disabled _replace_forwad =================')


LATENT_MEM = CrossFrameAttnLatentMemory()

script_callbacks.on_model_loaded(_replace_forwad)
script_callbacks.on_script_unloaded(unload)


DUMMY_IMAGE = "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"
