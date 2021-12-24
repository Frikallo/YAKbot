<h2 align="center">
  <img src="https://cdn.discordapp.com/attachments/882342184924348478/923980750615904296/image-modified.png" height='200px' width='200px'>
</h2>

<h1 align="center">BATbot🦇</h1>
<h4 align="center">A complex, multipurpose AI Discord bot.</h4>

<h1 align="center">
  <img src="https://img.shields.io/badge/discord.py-2.0-blue?style=flat" />
  <img src="https://img.shields.io/badge/Python-3.9-green?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/branch-development-red.svg" />
  <img src="https://img.shields.io/badge/build-passing-green.svg">
</h1>

## Info

BATbot is a collection of AI models based on image generation, analysis, and processing; all wrapped into one Discord Bot. BATbot's out-of-the-box commands range from image generation with VQGAN+CLIP or guided diffusion to image analysis and captioning with personality.

A full list of BATbots commands and uses can be found either in discord with the ```.help``` command, or right here:

|Command Syntax|Help|
|---|---
|`.rembg [Attatchment]`|**removes background from attached image**
|`.esrgan [Attatchment]`|**BATbot will use a pre-trained ESRGAN upscaler to upscale the resolution of your image up to 4 times**
|`.status`|**sends embed message with all relevant device stats for BATbot**
|`.imagine [Prompt]`|**uses CLIP+VQGAN open generation to create an original image from your prompt**
|`.diffusion [Prompt]`|**BATbot uses a CLIP+Diffusion model to generate images to match your prompt**
|`.facehq, .wikiart, .default, .d1024`|**Changes BATbots VQGAN+CLIP model to one trained solely on faces, art or default configuration**
|`.square, .landscape, .portrait`|**BATbot will update his size configurations for generations to your specified orientation**
|`.seed [Desired Seed]`|**Changes BATbots seed for all open generation (if 0 will set to random)**
|`.gptj [Prompt]`|**BATbot will use his trained GPT-J model to finish your prompt with natural language generation**
|`.sop [Attatchment]`|**BATbot will turn your attached image into a sequence of wave forms legible by a computer, this allows BATbot to create a sound correlating to the "sounds of processing"**
|`.faces [Attatchment]`|**BATbot will look through your photo and try to find any recognizable faces**

## Installation and Usage

First, install PyTorch 1.9.0 and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:
```bash
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
Replace cu111 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU. Next, we install the dependant packages:
* Guided Diffusion
```bash
$ git clone https://github.com/crowsonkb/guided-diffusion
$ pip install -e ./guided-diffusion
$ pip install lpips
$ curl -OL 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
```
* Personality-CLIP
```bash
$ git clone https://github.com/dzryk/cliptalk.git
$ cd cliptalk/
$ git clone https://github.com/dzryk/clip-grams.git
$ git clone https://github.com/openai/CLIP
$ pip install ftfy
$ pip install transformers
$ pip install autofaiss
$ pip install wandb
$ pip install webdataset
$ pip install git+https://github.com/PyTorchLightning/pytorch-lightning
$ curl -OL 'https://drive.google.com/uc?id=1fhWspkaOJ31JS91sJ-85y1P597dIfavJ'
$ curl -OL 'https://drive.google.com/uc?id=1PJcBni9lCRroFqnQBfOJOg9gVC5urq2H'
$ curl -OL 'https://drive.google.com/uc?id=13Xtf7SYplE4n5Q-aGlf954m6dN-qsgjW'
$ curl -OL 'https://drive.google.com/uc?id=1xyjhZMbzyI-qVz-plsxDOXdqWyrKbmyS'
$ curl -OL 'https://drive.google.com/uc?id=1peB-l-CWtwx0NKAIeAcwsnisjocc--66'
$ mkdir checkpoints
$ mkdir unigrams
$ mkdir bigrams
$ mkdir artstyles
$ mkdir emotions
$ unzip ./model.zip -d checkpoints
$ unzip ./unigrams.zip -d unigrams
$ unzip ./bigrams.zip -d bigrams
$ unzip ./artstyles.zip -d artstyles
$ unzip ./emotions.zip -d emotions
```
* VQGAN+CLIP(z+quantize)
```bash
$ git clone https://github.com/openai/CLIP
$ git clone https://github.com/CompVis/taming-transformers.git
$ pip install ftfy regex tqdm omegaconf pytorch-lightning
$ pip install kornia
$ pip install imageio-ffmpeg   
$ pip install einops          
$ mkdir steps
$ curl -L -o vqgan_imagenet_f16_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml' #ImageNet 1024
$ curl -L -o vqgan_imagenet_f16_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'  #ImageNet 1024
$ curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
$ curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
$ curl -L -o faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT' #FacesHQ
$ curl -L -o faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt' #FacesHQ
$ curl -L -o wikiart_16384.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml' #WikiArt 16384
$ curl -L -o wikiart_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt' #WikiArt 16384
```
Once installed just run:
```
pip install -r requirements.txt
```
Before running BATbot make sure you have your bot token set.
```python
#The end of your bot.py file should look something like this.
bot.run('qTIzNTA4NjMhUJI3NzgzJAAy.YcOCbw.GMYbjBWdiIWBPFrm_IMlUTlMGjM') #Your Token Here
```
Now finally run the bot:
```python
python3 bot.py
```
* Enjoy!

**Note: We do not provide support for self-hosting. If you are unable to self-host BATbot by yourself, just join [my discord server](https://discord.gg/KyU9tFN7gy) where BATbot runs 24/7.**

## Other repos

You may also be interested in <https://github.com/afiaka87/clip-guided-diffusion>

For upscaling images, try <https://github.com/xinntao/Real-ESRGAN>

## Citations

```python
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}

@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```
* Guided Diffusion - <https://github.com/openai/guided-diffusion> or [256x256](https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj) By Katherine Crowson
* Katherine Crowson - <https://github.com/crowsonkb>
* captions with personality - [colab notebook](https://colab.research.google.com/drive/171GirNbCVc-ScyBynI3Uy2fgYcmW3BB) from [dzyrk](https://github.com/dzryk)
* z+quantize notebook - [colab notebook](https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ) from [crimeacs](https://github.com/crimeacs)


## License
[MIT](https://github.com/Frikallo/BATbot/blob/main/LICENSE)
