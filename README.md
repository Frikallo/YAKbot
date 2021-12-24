# BATbot

BATbot is a collection of AI models based on image generation, analysis, and processing; all wrapped into one Discord Bot. BATbot's out-of-the-box commands range from image generation with VQGAN+CLIP or guided diffusion to image analysis and captioning with personality.

A full list of BATbots commands and uses can be found either in discord with the ```.help``` command, or can be found right here:

|Command Syntax|Help|
|---|---
|.rembg [Attached Image]|**removes background from attached image**
|.esrgan [Attatchment]|**BATbot will use a pretrained ESRGAN upscaler to upscale you images resolution by up to 4 times**

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
```
* Guided Diffusion - <https://github.com/openai/guided-diffusion> or [256x256](https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj) By Katherine Crowson
* Katherine Crowson - <https://github.com/crowsonkb>
* captions with personality - [colab notebook](https://colab.research.google.com/drive/171GirNbCVc-ScyBynI3Uy2fgYcmW3BB) from [dzyrk](https://github.com/dzryk)
* z+quantize notebook - [colab notebook](https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ) from [crimeacs](https://github.com/crimeacs)


## License
[MIT](https://github.com/Frikallo/BATbot/blob/main/LICENSE)