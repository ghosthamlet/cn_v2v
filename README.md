
# ControlNet only video2video webui custom script

A simple script for preset optimized ControlNets settings in code to convert video to cartoon, just change few options, you can convert arbitary video (except grayscale or cartoon video) with no length limit to cartoon, not just Dancing girl videos but all type videos.

No optical flow, no ebsynth, no postprocessing, very stable background, sometimes stable humans and faces.


See a 6min converted video: https://youtu.be/lAN_ziOZCfQ

If you want to know all the settings for convert the above video, it is here:

prompt: Ink style

Negative prompt: deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck, lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature

Steps: 6, Sampler: Euler, CFG scale: 1, Seed: 1259467077, Size: 960x512, Model hash: fded6ea807, Model: animelike25D_animelike25DPruned, Denoising strength: 0.8, Version: v1.2.1, ControlNet 0: "preprocessor: lineart_realistic, model: control_v11p_sd15_lineart_fp16 [5c23b17d], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: ControlNet is more important, preprocessor params: (512, 64, 64)", ControlNet 1: "preprocessor: softedge_hedsafe, model: control_v11p_sd15_softedge_fp16 [f616a34f], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: ControlNet is more important, preprocessor params: (512, 1, 64)", ControlNet 2: "preprocessor: depth_midas, model: control_v11f1p_sd15_depth_fp16 [4b72d323], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: ControlNet is more important, preprocessor params: (512, 64, 64)", ControlNet 3: "preprocessor: normal_bae, model: control_v11p_sd15_normalbae_fp16 [592a19d8], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: ControlNet is more important, preprocessor params: (512, 64, 64)", ControlNet 4: "preprocessor: none, model: diff_control_sd15_temporalnet_fp16 [adc6bd97], weight: 0.5, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: Balanced, preprocessor params: (512, 1, 64)", ControlNet 5: "preprocessor: tile_resample, model: control_v11f1e_sd15_tile_fp16 [3b860298], weight: 1, starting/ending: (0, 1), resize mode: Crop and Resize, pixel perfect: True, control mode: ControlNet is more important, preprocessor params: (512, 1, 64)"

Most of these settings is preseted in cn_v2v code.


### Requirements

Only tested on Ubuntu. 

Install ffmpeg. 

Install controlnet webui extension: https://github.com/Mikubill/sd-webui-controlnet (download models: control_v11p_sd15_lineart_fp16, control_v11p_sd15_softedge_fp16, control_v11f1p_sd15_depth_fp16, control_v11p_sd15_normalbae_fp16, control_v11f1e_sd15_tile_fp16, and diff_control_sd15_temporalnet_fp16 from https://huggingface.co/CiaraRowles/TemporalNet)

Download checkpoint animelike25D_animelike25DPruned [fded6ea807] from civitai.com.

Copy cn_v2v.py to webui scripts/ folder to install it.

Enable xformers, with xformers, max GPU VRAM usage is 10800MiB, for 7s video (width/height: 960/512) convert time is 11min on 4090.

In webui settings, change max controlnet unit to 6.
         

### Usage

As the checkpoint downloaded in Requirements animelike25D_animelike25DPruned is very good for cartoon, so no need to give style words in prompt, and presets in cn_v2v have cfg_scale=1, so change prompt will have few effects, just input `cartoon` for all videos is ok, and negative_prompt is preseted in code, you can leave it empty.

#### simple method
In img2img panel, Change width/height, select `CN v2v` in script dropdown, upload a video, wait until it upload fininsh, there will be a 'Download' link. 

After that, you can see two links appeared at the page bottom, the first link is the first frame image of converted video, the second link is the converted video, after convert finished, you can click the two links to check them. (I still don't know simple method to show video after convert finished in gradio)

(optional, if you want more cartoon, change Temporalnet weight to 0.2, but it will have more flickering)

Settings finished, go click Generate.

#### improve quality
you can just change the seed, different seed sometimes have very different quality in details.

For test seed, you can upload a important video frame in img2img, and enable `Test settings on a single image, have to upload a image in img2img` in cn_v2v.  

Settings finished, go click Generate.

After you find a better seed, fill it in Seed input, disable `Test settings on a single image, have to upload a image in img2img`, follow above `simple method` to convert video. 
