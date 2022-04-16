<h2 align="center">
  <img src="https://cdn.discordapp.com/attachments/882342184924348478/956735304105066517/1648173184_Photograph_Portrait_Yak_staring_into_the_camera_by_Studio_Ghibli_-hq-modified.png" height='250px' width='250px'>
</h2>

<h1 align="center">YAKbot🐂</h1>
<h4 align="center">A complex, multipurpose AI Discord bot.</h4>

<h1 align="center">
  <img src="https://img.shields.io/badge/stage-development-yellow.svg" />
  <img src="https://img.shields.io/lgtm/alerts/g/Frikallo/BATbot.svg?logo=lgtm&logoWidth=18)">
  <img src="https://img.shields.io/codeclimate/maintainability-percentage/Frikallo/BATbot?logo=codeclimate&color=pine">
  <img src="https://img.shields.io/codeclimate/maintainability/Frikallo/BATbot?label=code%20quality&logo=codeclimate&color=pine">
  <img src="https://github.com/frikallo/batbot/actions/workflows/main.yml/badge.svg">
</h1>

## Info
[[Info]](https://github.com/Frikallo/YAKbot#info) [[Environment]](https://github.com/Frikallo/YAKbot#environment) [[Setup & Install]](https://github.com/Frikallo/YAKbot#setup-and-installation) [[Citations]](https://github.com/Frikallo/YAKbot#other-repos)

YAKbot is a collection of AI models based on image generation, analysis, and processing; all wrapped into one Discord Bot. YAKbot's out-of-the-box commands range from image generation with VQGAN+CLIP or guided diffusion to image analysis and captioning with personality.

A full list of YAKbots commands and uses can be found either in discord with the ```.help``` command, or right here:

|Command Syntax|Help|Examples|
|---|---|---
|`.rembg [Attatchment]`|**removes background from attached image**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.rembg/rembg.jpeg)
|`.esrgan [Attatchment]`|**YAKbot will use a pre-trained ESRGAN upscaler to upscale the resolution of your image up to 4 times**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.esrgan/esrgan.jpeg)
|`.status`|**sends embed message with all relevant device stats for YAKbot**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.status/status.jpeg)
|`.imagine [Prompt]`|**YAKbot will use CLIP+VQGAN open generation to create an original image from your prompt**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.imagine/imagine1.jpeg)
|`.facehq, .wikiart, .default, .d1024`|**Changes YAKbots VQGAN+CLIP model to one trained solely on faces, art or default configuration**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.facehq%2C%20.wikiart%2C%20.default%2C%20.d1024/facehq_wikiart_default_deafault1024.jpeg)
|`.square, .landscape, .portrait`|**YAKbot will update his size configurations for generations to your specified orientation**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.square%2C%20.landscape%2C%20.portrait/square_landscape_portrait.jpeg)
|`.seed [Desired Seed]`|**Changes YAKbots seed for all open generation (if 0 will set to random)**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.seed/seed.jpeg)
|`.faces [Attatchment]`|**YAKbot will look through your photo and try to find any recognizable faces**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.ld/latentdiffusion.jpeg)
|`.latentdiffusion [Prompt] or .ld [Prompt]`|**This command is another Text2Image method like `.imaging` but uses a method called latent diffusion. innitiate this method by using the command**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.colorize/colorize.jpeg)
|`.outline [Prompt]`|**YAKbot will contact a local GPT3 model that will synthasize and look for essays on your prompt while outputting an outline/list of ideas/facts about your prompt to help kickstart your projects**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/.outline/outline.jpeg)
|`[Attatchment]`|**YAKbot will first decide if your attatchment is neutral, negative or positive and then based on that will try to caption your image with both text, and emoji**|[Example](https://github.com/Frikallo/YAKbot/blob/main/Bot/Examples/CLIPCaption/CLIPcaption1.jpeg)

To see examples of all the different commands click here: [Examples](https://github.com/Frikallo/YAKbot/tree/main/Bot/Examples)


## Environment
* Windows 11 (10.0.22)
* Anaconda
* Nvidia RTX 2070 Super (8GB)

Typical VRAM requirements:

* VQGAN(256x256 - 512x512) **~** `5-10GB`
* Diffusion 256x256 **~** `6-7GB`
* Diffusion 512x256 **~** `10-12GB`
* Image classification & captioning **~** `4GB`

## Setup and Installation

First, install PyTorch 1.9.0 and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:
```bash
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
Replace cu111 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU. Next, we install the dependant packages:

<details>
  <summary>Install Instructions (Click me)</summary>
  <!-- have to be followed by an empty line! -->
  
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
$ unzip ./model.zip -d checkpoints #make sure the unzipped "model" Folder goes in ./YAKbot/Bot/checkpoints 
$ unzip ./unigrams.zip -d unigrams #make sure the unzipped "unigrams" Folder goes in ./YAKbot/Bot/unigrams 
$ unzip ./bigrams.zip -d bigrams #make sure the unzipped "bigrams" Folder goes in ./YAKbot/Bot/bigrams 
$ unzip ./artstyles.zip -d artstyles #make sure the unzipped "artstyles" Folder goes in ./YAKbot/Bot/artstyles 
$ unzip ./emotions.zip -d emotions #make sure the unzipped "emotions" Folder goes in ./YAKbot/Bot/emotions 
```
</details>

* **Personality-CLIP**
<details>
  <summary>Install Instructions (Click me)</summary>
  <!-- have to be followed by an empty line! -->

```bash
$ git clone https://github.com/openai/CLIP
$ git clone https://github.com/CompVis/taming-transformers.git
$ pip install ftfy regex tqdm omegaconf pytorch-lightning
$ pip install kornia
$ pip install imageio-ffmpeg   
$ pip install einops          
$ mkdir steps
#place all of the following model files in ./YAKbot/Bot
$ curl -L -o vqgan_imagenet_f16_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml' #ImageNet 1024
$ curl -L -o vqgan_imagenet_f16_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'  #ImageNet 1024
$ curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
$ curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
$ curl -L -o faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT' #FacesHQ
$ curl -L -o faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt' #FacesHQ
$ curl -L -o wikiart_16384.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml' #WikiArt 16384
$ curl -L -o wikiart_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt' #WikiArt 16384
```
</details>

* **VQGAN+CLIP(z+quantize)**

Once all are installed just run:
```
$ git clone https://github.com/Frikallo/YAKbot.git
$ cd YAKbot 
$ pip install -r requirements.txt
```
Before YAKbot can start make sure you have your bot token set.
```python
#The end of your bot.py file should look something like this.
bot.run('qTIzNTA4NjMhUJI3NzgzJAAy.YcOCbw.GMYbjBWdiIWBPFrm_IMlUTlMGjM') #Your Token Here
```
Now finally run the bot:
```
$ cd Bot
$ python3 bot.py
```
* Enjoy!

**Note: I will not provide support for self-hosting. If you are unable to self-host YAKbot by yourself, just join [my discord server](https://discord.gg/KyU9tFN7gy) where YAKbot runs 24/7.**

A successful startup will look like this:
<details>
  <summary>(Click me)</summary>
  <!-- have to be followed by an empty line! -->

  <h1 align="center">
  <img src="https://cdn.discordapp.com/attachments/924037023315165254/930992116056858704/Screenshot_2022-01-12_170940.png">
  </h1>
</details>

## .env Setup

If self hosting, make sure you have a .env file within the ./Bot directory. Your evironment file should look somthing like this:
```bash
"YAKbot/Bot/.env"
prompt = '--'
seed = '42'
infile = '--'
model = 'vqgan_imagenet_f16_16384' #Model checkpoint file for .imagine
width = '412' #Width in pixels for .imagine
height = '412' #Height in pixels for .imagine
max_iterations = '400' #Max iterations for .imagine
bot_token = '--' #Discord bot token, get yours at https://discord.com/developers/applications
input_image = '--'
CB1 = 'True'
CB2 = 'True'
CB3 = 'True'
webhook = '--' #Discord webhook url for startup info
OPENAI_KEY = '--' #OpenAI API token for GPT3 generation commands
```

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
* Guided Diffusion - <https://github.com/openai/guided-diffusion>


* Original 256x256 notebook: [![Open In Colab][colab-badge]][colab-notebook1] from [Katherine Crowson](https://github.com/crowsonkb)

[colab-notebook1]: <https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj#scrollTo=X5gODNAMEUCR>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
* Original 512x512 notebook: [![Open In Colab][colab-badge]][colab-notebook2] from [Katherine Crowson](https://github.com/crowsonkb)

[colab-notebook2]: <https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA#scrollTo=VnQjGugaDZPJ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

* Original Caption notebook: [![Open In Colab][colab-badge]][colab-notebook3] from [dzyrk](https://github.com/dzryk)

[colab-notebook3]: <https://colab.research.google.com/drive/171GirNbCVc-ScyBynI3Uy2fgYcmW3BB>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

* Original z+quantize notebook: [![Open In Colab][colab-badge]][colab-notebook3] from [crimeacs](https://github.com/crimeacs)

[colab-notebook3]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

**This project was HEAVILY influenced and inspired by [EleutherAI](https://eleuther.ai)'s discord bot (BATbot)**

## License
```
MIT License

Copyright (c) 2021 Frikallo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
