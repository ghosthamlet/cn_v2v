
# ControlNet only video2video webui custom script

A simple script for preset ControlNet in code to convert video to cartoon, just change few options, you can convert arbitary video (except grayscale or cartoon video) with no length limit to cartoon.

No optical flow, no ebsynth, no postprocessing, very stable background, sometimes stable humans and faces.

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
you can just change the seed, different seed sometimes have very different quality.
For test seed, you can upload a important video frame in img2img, and enable `Test settings on a single image, have to upload a image in img2img` in cn_v2v.  
Settings finished, go click Generate.
After you find a better seed, fill it in Seed input, disable `Test settings on a single image, have to upload a image in img2img`, follow above `simple method` to convert video. 
