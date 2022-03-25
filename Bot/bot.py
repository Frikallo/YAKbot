import time
from clip.clip import available_models
from multiprocessing import Process

from sqlalchemy import true

startTime = time.time()
import threading
from discord_webhook import DiscordWebhook, DiscordEmbed
from time import localtime, strftime
import argparse
import socket
from pyston import PystonClient, File
import asyncio
import PySimpleGUI as sg
import presets
import argparse
import matplotlib.pyplot as plt
from colorizers import *
import sys
from upscaler import upscale
import urllib.request
import datetime
import traceback
from moviepy.editor import VideoFileClip
from music21 import instrument, note, chord, stream
import moviepy.video.fx.all as vfx
from discord.ext import commands
import face_recognition
from PIL import Image, ImageDraw
from discord import Embed
import statcord
from discord import FFmpegPCMAudio
import subprocess
import torch
from win32com.client import GetObject
from torch._C import wait
import torch.nn as nn
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import discord
from classify import load, classify, encode
import os

os.chdir("./Bot")
print(os.getcwd())
import random
import asyncio
from io import BytesIO
import pandas as pd
import json
import gc
import psutil

import clip
import glob
from pytorch_lightning.callbacks import ModelCheckpoint

import model
import utils

import faiss
import requests
import torchvision.transforms.functional as F

from torchvision.utils import make_grid

import model
import retrofit

from rembg.bg import remove
import numpy as np
import io
from PIL import ImageFile

import openai
import math

from vqgan_clip.grad import *
from vqgan_clip.helpers import *
from vqgan_clip.inits import *
from vqgan_clip.masking import *
from vqgan_clip.optimizers import *

start = time.time()

from urllib.request import urlopen
from tqdm import tqdm

from omegaconf import OmegaConf

from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
import datetime as dt

torch.backends.cudnn.benchmark = False

from torch_optimizer import DiffGrad, AdamP, RAdam

import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops

ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re
from packaging import version

# Supress warnings
import warnings

warnings.filterwarnings("ignore")

# For Later Tinkering
# import vclip
# from vclip import vclip as vc

ImageFile.LOAD_TRUNCATED_IMAGES = True
CB1 = os.environ.get("CB1")
CB2 = os.environ.get("CB2")
CB3 = os.environ.get("CB3")
Indices_value = "False"
CLIP = "False"
openai.api_key = os.environ.get("OPENAI_KEY")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

answer = input("Load CLIP? (y/n) ")
if answer == "y":
    CLIP = "True"
    print("Loading Models...")
    model2, preprocess2 = clip.load("ViT-B/32", device=device)
    text2 = clip.tokenize(["negative", "neutral", "positive"]).to(device)
    print("Using device:", device)
    load_categories = "emojis"
    load(load_categories)
    print("Clip loaded.")
else:
    print("Suppressing Clip Models")

lowerBoundNote = 21
resolution = 0.25
MAIN_COLOR = 0x459FFF  # light blue kinda

use_gpu = False
save_prefix = "saved"

# filepaths
fp_in = ".\*.png"
fp_out = "movie.mp4"

# face recognition settings
# Noah
image_of_noah = face_recognition.load_image_file("./img/known/Noah.png")
noah_face_encoding = face_recognition.face_encodings(image_of_noah)[0]
# Jack
image_of_jack = face_recognition.load_image_file("./img/known/Jack.png")
jack_face_encoding = face_recognition.face_encodings(image_of_jack)[0]
# Ewan
image_of_ewan = face_recognition.load_image_file("./img/known/Ewan.png")
ewan_face_encoding = face_recognition.face_encodings(image_of_ewan)[0]
# Erin
image_of_erin = face_recognition.load_image_file("./img/known/Erin.png")
erin_face_encoding = face_recognition.face_encodings(image_of_erin)[0]
# Brent
image_of_brent = face_recognition.load_image_file("./img/known/Brent.png")
brent_face_encoding = face_recognition.face_encodings(image_of_brent)[0]
# Connor
image_of_connor = face_recognition.load_image_file("./img/known/Connor.png")
connor_face_encoding = face_recognition.face_encodings(image_of_connor)[0]
# Liam
image_of_liam = face_recognition.load_image_file("./img/known/Liam.png")
liam_face_encoding = face_recognition.face_encodings(image_of_liam)[0]
# Alek
image_of_alek = face_recognition.load_image_file("./img/known/Alek.png")
alek_face_encoding = face_recognition.face_encodings(image_of_alek)[0]
# Obama
image_of_obama = face_recognition.load_image_file("./img/known/Obama.png")
obama_face_encoding = face_recognition.face_encodings(image_of_obama)[0]

#  Create arrays of encodings and names
known_face_encodings = [
    noah_face_encoding,
    jack_face_encoding,
    ewan_face_encoding,
    erin_face_encoding,
    brent_face_encoding,
    connor_face_encoding,
    liam_face_encoding,
    alek_face_encoding,
    obama_face_encoding,
]

known_face_names = [
    "Noah",
    "Jack",
    "Ewan",
    "Erin",
    "Brent",
    "Connor/Ari",
    "Liam",
    "Alek",
    "Obama",
]

# Helper functions
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def display_grid(imgs):
    reshaped = [F.to_tensor(x) for x in imgs]
    show(make_grid(reshaped))


def load_image(img, preprocess):
    img = Image.open(fetch(img))
    return img, preprocess(img).unsqueeze(0).to(device)


def clip_rescoring(args, net, candidates, x):
    textemb = net.perceiver.encode_text(
        clip.tokenize(candidates).to(args.device)
    ).float()
    textemb /= textemb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * x @ textemb.T).softmax(dim=-1)
    _, indices = similarity[0].topk(args.num_return_sequences)
    return [candidates[idx] for idx in indices[0]]


def fetch(url_or_path):
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def caption_image(path, args, net, preprocess, context):
    captions = []
    img, mat = load_image(path, preprocess)
    table, x = net.build_table(
        mat.half(),
        net.perceiver,
        ctx=context,
        indices=net.indices,
        indices_data=net.indices_data,
        knn=args.knn,
        tokenize=clip.tokenize,
        device=args.device,
        is_image=True,
        return_images=True,
    )

    table = net.tokenizer.encode(table[0], return_tensors="pt").to(device)
    table = table.squeeze()[:-1].unsqueeze(0)
    out = net.model.generate(
        table,
        max_length=args.maxlen,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
    )
    candidates = []
    for seq in out:
        decoded = net.tokenizer.decode(seq, skip_special_tokens=True)
        decoded = decoded.split("|||")[1:][0].strip()
        candidates.append(decoded)
    captions = clip_rescoring(args, net, candidates, x[None, :])
    print(f"Personality: {context[0]}\n")
    for c in captions[: args.display]:
        print(c)
    display_grid([img])
    return captions


def success_embed(title, description):
    return Embed(title=title, description=description, color=MAIN_COLOR)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


async def play_source(voice_client):
    source = FFmpegPCMAudio("krusty_krab_theme.mp3")
    voice_client.play(
        source,
        after=lambda e: print("Player error: %s" % e)
        if e
        else bot.loop.create_task(play_source(voice_client)),
    )


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def hms(seconds):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)


# Settings
args = argparse.Namespace(
    config="./checkpoints/12xdqrwd-config",
    index_dirs="./unigrams,./bigrams,./artstyles,./emotions",
    clip_model="ViT-B/16",
    knn=3,
    maxlen=72,
    num_return_sequences=32,  # decrease this is you get GPU OOM, increase if using float16 model
    num_beams=1,
    temperature=0.8,
    top_p=0.9,
    display=1,
    do_sample=True,
    device=device,
)

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

# Load indices
answer = input("Load indices? (y/n) ")

if answer == "y":
    Indices_value = "True"
    print("Loading indices...")
    indices = []
    indices_data = []
    index_dirs = args.index_dirs.split(",")
    index_dirs = list(filter(lambda t: len(t) > 0, index_dirs))
    for index_dir in index_dirs:
        fname = os.path.join(index_dir, "args.txt")
        with open(fname, "r") as f:
            index_args = dotdict(json.load(f))

        entries = []
        fname = os.path.join(index_dir, "entries.txt")
        with open(fname, "r") as f:
            entries.extend([line.strip() for line in f])

        indices_data.append(entries)
        indices.append(faiss.read_index(glob.glob(f"{index_dir}/*.index")[0]))
    preprocess = clip.load(args.clip_model, jit=False)[1]

    # Load model
    config = dotdict(torch.load(args.config))
    config.task = "txt2txt"
    config.adapter = "./checkpoints/12xdqrwd.ckpt"
    net = retrofit.load_params(config).to(device)
    net.indices = indices
    net.indices_data = indices_data
    print("Loaded indices.")
else:
    print("Not loading indices")

# Bot
bot = commands.Bot(command_prefix=".", help_command=None)

key = "statcord.com-pYzEX1uxAJv7PAT4ZaAx"
api = statcord.Client(bot, key)
api.start_loop()

dev_id = 882342184924348478
public_id = 920889454443524116
answer = input("Use dev channel? (y/n) ")
if answer == "y":
    channel_id = dev_id
else:
    id = input("Channel ID: ")
    if id == "n":
        public_id = 920889454443524116
    else:
        public_id = int(id)
    channel_id = public_id


@bot.event
async def on_ready():
    await bot.change_presence(
        status=discord.Status.online, activity=discord.Game(name=f"in a trash bin")
    )
    print("We have logged in as {0.user}".format(bot))
    print(f"Connected to: {len(bot.guilds)} guilds")
    # for guild in bot.guilds:
    #    print(f"{guild.name} ({guild.id})")
    print(f"Connected to: {len(bot.commands)} commands")
    YAKbot = """██╗░░░██╗░█████╗░██╗░░██╗██████╗░░█████╗░████████╗
╚██╗░██╔╝██╔══██╗██║░██╔╝██╔══██╗██╔══██╗╚══██╔══╝
░╚████╔╝░███████║█████═╝░██████╦╝██║░░██║░░░██║░░░
░░╚██╔╝░░██╔══██║██╔═██╗░██╔══██╗██║░░██║░░░██║░░░
░░░██║░░░██║░░██║██║░╚██╗██████╦╝╚█████╔╝░░░██║░░░
░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═════╝░░╚════╝░░░░╚═╝░░░"""
    print(YAKbot)
    # print(f'Loaded cogs: {bot.cogs}')
    end = time.time() - startTime
    end = int(end)

    endtime = hms(end)
    print(f"Elapsed Startup Time: {endtime}")
    date = strftime("%a, %d %b %Y %H:%M:%S", localtime())
    print(date)
    webhook = DiscordWebhook(url=os.environ["webhook"],)
    output = f"""We have logged in as YAKbot#7261
Connected to: {len(bot.guilds)} guilds
Connected to: {len(bot.commands)} commands

__  _____    __ __ __          __ 
\ \/ /   |  / //_// /_  ____  / /_
 \  / /| | / ,<  / __ \/ __ \/ __/
 / / ___ |/ /| |/ /_/ / /_/ / /_  
/_/_/  |_/_/ |_/_.___/\____/\__/  
                                  
                                         
Elapsed Startup Time: {endtime}
{date}"""
    r = requests.get(
        "https://discord.com/api/webhooks/935708137682530364/w74pfb3cnr5JnkOaIPv73Y0f-ast4ygNBUDQ3_wpbxaF_z5_fe0U1HWGsnp4TLvm57l6"
    )
    response = str(r.elapsed.microseconds)
    response = response[0:2]
    embed = DiscordEmbed(
        title="Output", description=f"```{output}```", color="0x7289DA"
    )
    embed.set_author(
        name="",
        url="https://cdn.discordapp.com/attachments/935708113439436841/935778588173688913/88942100.png",
    )
    embed.set_footer(text="V.1.29")
    embed.set_timestamp()
    embed.add_embed_field(
        name="Startup Stats",
        value=f"CLIP? **{CLIP}**\n Indices? **{Indices_value}**\nStartup Time: `{endtime}`",
        inline=True,
    )
    devicename = str(socket.gethostname())
    if bot:
        success = "True"
    else:
        success = "False"
    success = "True"
    embed.add_embed_field(
        name="Connection Stats",
        value=f"Response Time: `({response}ms)`\nTriggered by: `({devicename})`\nSuccessful startup? `({success})`",
        inline=True,
    )
    webhook.add_embed(embed)
    response = webhook.execute()


@bot.listen("on_message")
async def image(ctx):
    if ctx.author.bot:
        return
    if (
        ctx.attachments
        and ctx.content != ".rembg"
        and ctx.content != ".faces"
        and ctx.content != ".esrgan"
        and ctx.content != ".colorize"
        and ctx.content != ".imagine"
    ):
        if ctx.channel.id != channel_id:
            return
        gc.collect()
        torch.cuda.empty_cache()
        async with ctx.channel.typing():
            await asyncio.sleep(0.1)
            # Download Image
            link = ctx.attachments[0].url
            filename = link.split("/")[-1]
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)

            # Caption+Emoji
            file_ = filename
            print("classifying")
            image2 = preprocess2(Image.open(file_)).unsqueeze(0).to(device)
            try:
                reaction = classify(file_)
                reaction = str(reaction)
                print(reaction)
                emoji = f"{reaction}"
                emoji = emoji.encode("unicode-escape").decode("ASCII")
                with open("reactables.txt", "a+") as file_object:
                    file_object.write(emoji)
                with open("reactables.txt") as f:
                    first_line = f.readline()
                os.remove("reactables.txt")
                first_line = first_line[:-2]
                first_line = (
                    first_line.encode("ascii")
                    .decode("unicode-escape")
                    .encode("utf-16", "surrogatepass")
                    .decode("utf-16")
                )
                first_line = f"{first_line}"
                os.remove(filename)
                print(ctx.attachments[0].url)
                img = ctx.attachments[0].url

                with torch.no_grad():
                    image_features = model2.encode_image(image2)
                    text_features = model2.encode_text(text2)

                    logits_per_image, logits_per_text = model2(image2, text2)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                Neg = []
                Neu = []
                Pos = []

                length = len(probs)
                middle_index = length // 3
                listA = probs[middle_index]
                Neg.append(listA[0])
                Neu.append(listA[1])
                Pos.append(listA[2])

                strings_Neg = [str(integer) for integer in Neg]
                Neg = "".join(strings_Neg)
                Neg = float(Neg)

                strings_Neu = [str(integer) for integer in Neu]
                Neu = "".join(strings_Neu)
                Neu = float(Neu)

                strings_Pos = [str(integer) for integer in Pos]
                Pos = "".join(strings_Pos)
                Pos = float(Pos)

                Neg_Captions = [
                    "dread",
                    "hurt",
                    "suffering",
                    "afraid",
                    "aggressive",
                    "alarmed",
                    "annoyed",
                    "anxious",
                    "bitter",
                    "brooding",
                    "claustrophobic",
                    "cowardly",
                    "demoralized",
                    "depressed",
                    "disheartened",
                    "disoriented",
                    "dispirited",
                    "disturbed",
                    "frustrated",
                    "gloomy",
                    "helpless",
                    "hopeless",
                    "horrified",
                    "impatient",
                    "indignant",
                    "infuriated",
                    "irritated",
                    "lonely",
                    "mad",
                    "miserable",
                    "mortified",
                    "nasty",
                    "nauseated",
                    "negative",
                    "nervous",
                    "offended",
                    "panicked",
                    "paranoid",
                    "possessive",
                    "powerless",
                    "rash",
                    "rejected",
                    "scared",
                    "shocked",
                    "smug",
                    "stressed",
                    "stubborn",
                    "stuck",
                    "suspicious",
                    "troubled",
                    "unhappy",
                    "unsettled",
                    "unsure",
                    "upset",
                    "vengeful",
                    "vicious",
                    "vulnerable",
                    "weak",
                    "worried",
                    "bad",
                    "bitter",
                    "cold",
                    "crazydead",
                    "dry",
                    "fat",
                    "hollow",
                    "old",
                    "plain",
                    "poor",
                    "shy",
                    "sore",
                    "sour",
                    "wrong",
                    "dark",
                    "shadow",
                    "anger",
                    "anxiousness",
                    "cruelty",
                    "cynic",
                    "denial",
                    "depression",
                    "despair",
                    "disgust",
                    "emptiness",
                    "fear",
                    "gray",
                    "grief",
                    "hate",
                    "hostile",
                    "hurt",
                    "indignation",
                    "insanity",
                    "irritability",
                    "jealousy",
                    "longing",
                    "lust",
                    "mourning",
                    "needy",
                    "pain",
                    "paranoia",
                    "pity",
                    "possessive",
                    "pride",
                    "rage",
                    "remorse",
                    "resentment",
                    "resignation",
                    "sadness",
                    "scorn",
                    "sensitive",
                    "shame",
                    "sorrow",
                    "tense",
                    "uncertainty",
                    "uneasiness",
                    "upset",
                    "agitation",
                    "agony",
                    "alarm",
                    "alienation",
                    "anguish",
                    "apathy",
                    "aversion",
                    "brooding",
                    "confusion",
                    "cynicism",
                    "dejection",
                    "disappointment",
                    "disbelief",
                    "discomfort",
                    "discontentment",
                    "displeasure",
                    "distraction",
                    "distress",
                    "doubt",
                    "dread",
                    "embarrassment",
                    "expectancy",
                    "fright",
                    "fury",
                    "glumness",
                    "greed",
                    "grumpiness",
                    "guilt",
                    "hatred",
                    "homesickness",
                    "humiliation",
                    "insecurity",
                    "loathing",
                    "miserliness",
                    "negative",
                    "neglect",
                    "outrage",
                    "paranoid",
                    "pessimism",
                    "rash",
                    "regret",
                    "restlessness",
                    "self-pity",
                    "spite",
                    "suffering",
                    "sullenness",
                    "suspense",
                    "tension",
                    "terror",
                    "woe",
                    "wrath",
                ]
                Neu_Captions = [
                    "arrogant",
                    "assertive",
                    "astonished",
                    "baffled",
                    "bewildered",
                    "bored",
                    "brazen",
                    "cheeky",
                    "coercive",
                    "content",
                    "dazed",
                    "determined",
                    "discombobulated",
                    "disgruntled",
                    "dominant",
                    "dumbstruck",
                    "exasperated",
                    "flakey",
                    "hospitable",
                    "insightful",
                    "isolated",
                    "lazy",
                    "loopy",
                    "moody",
                    "mystified",
                    "numb",
                    "obstinate",
                    "perplexed",
                    "persevering",
                    "pleased",
                    "puzzled",
                    "rattled",
                    "reluctant",
                    "ruthless",
                    "self-conscious",
                    "submissive",
                    "tired",
                    "unnerved",
                    "alert",
                    "beige",
                    "clear",
                    "elderly",
                    "few",
                    "giant",
                    "granite",
                    "petite",
                    "quiet",
                    "shallow",
                    "sharp",
                    "solid",
                    "square",
                    "swift",
                    "wet",
                    "wild",
                    "pale",
                    "flamboyant",
                    "neutral",
                    "animosity",
                    "anticipation",
                    "camaraderie",
                    "cautious",
                    "content",
                    "poem",
                    "down",
                    "envy",
                    "expectation",
                    "infatuation",
                    "interest",
                    "irritability",
                    "longing",
                    "lust",
                    "mean",
                    "mercy",
                    "mildness",
                    "perturbation",
                    "pity",
                    "resignation",
                    "sensitive",
                    "tense",
                    "uncertainty",
                    "uneasiness",
                    "yearning",
                    "ambivalence",
                    "apathy",
                    "apprehension",
                    "attentiveness",
                    "aversion",
                    "baffled",
                    "confusion",
                    "curiosity",
                    "dismay",
                    "distraction",
                    "dominant",
                    "epiphany",
                    "expectancy",
                    "fascination",
                    "hope",
                    "humility",
                    "hysteria",
                    "idleness",
                    "indifference",
                    "jubilation",
                    "melancholy",
                    "modesty",
                    "patience",
                    "restlessness",
                    "revulsion",
                    "self-pity",
                    "sentimentality",
                    "serenity",
                    "suspense",
                    "tension",
                ]
                Pos_Captions = [
                    "Ridiculous",
                    "caring",
                    "melancholy",
                    "calm",
                    "carefree",
                    "careless",
                    "comfortable",
                    "confident",
                    "delighted",
                    "driven",
                    "enchanted",
                    "enlightened",
                    "focused",
                    "kind",
                    "nostalgic",
                    "optimistic",
                    "positive",
                    "relaxed",
                    "relieved",
                    "self-confident",
                    "self-respecting",
                    "shameless",
                    "thrilled",
                    "triumphant",
                    "worthy",
                    "better",
                    "cool",
                    "fancy",
                    "good",
                    "light",
                    "modern",
                    "rich",
                    "safe",
                    "superior",
                    "sweet",
                    "tan",
                    "whispering",
                    "wise",
                    "nice",
                    "young",
                    "delicious",
                    "sweet",
                    "bold",
                    "sunny",
                    "flaming",
                    "gay",
                    "sparkling",
                    "shining",
                    "glitter",
                    "glowing",
                    "well",
                    "showing",
                    "bright",
                    "acceptance",
                    "adoration",
                    "affection",
                    "attraction",
                    "intellectual",
                    "spiritual",
                    "bliss",
                    "bubbly",
                    "calm",
                    "contempt",
                    "desire",
                    "eager",
                    "enlightened",
                    "exuberance",
                    "fulfillment",
                    "hopeful",
                    "peace",
                    "innocent",
                    "interest",
                    "joy",
                    "kind",
                    "longing",
                    "love",
                    "lust",
                    "melancholic",
                    "mercy",
                    "passion",
                    "pleasure",
                    "pride",
                    "relief",
                    "sensitive",
                    "sincerity",
                    "trust",
                    "yearning",
                    "admiration",
                    "amazement",
                    "ambivalence",
                    "amusement",
                    "attentiveness",
                    "baffled",
                    "sweetness",
                    "awe",
                    "caring",
                    "charity",
                    "cheerfulness",
                    "courage",
                    "curiosity",
                    "eagerness",
                    "ecstasy",
                    "empathy",
                    "enjoyment",
                    "enthusiasm",
                    "epiphany",
                    "euphoria",
                    "excitement",
                    "fascination",
                    "fondness",
                    "friendliness",
                    "glee",
                    "gratitude",
                    "happiness",
                    "hope",
                    "humility",
                    "liking",
                    "melancholy",
                    "modesty",
                    "patience",
                    "politeness",
                    "positive",
                    "satisfaction",
                    "sentimentality",
                    "surprise",
                    "sympathy",
                    "tenderness",
                    "thankfulness",
                    "tolerance",
                    "worthy",
                ]

                simple_Captions1 = ["negative"]
                simple_Captions2 = ["neutral"]
                simple_Captions3 = ["positive"]

                if Neg > Neu and Neg > Pos:
                    print("Negative")
                    caption_person = random.choice(Neg_Captions)
                    simple_person = random.choice(simple_Captions1)
                elif Neu > Neg and Neu > Pos:
                    print("Neutral")
                    caption_person = random.choice(Neu_Captions)
                    simple_person = random.choice(simple_Captions2)
                elif Pos > Neg and Pos > Neu:
                    print("Positive")
                    caption_person = random.choice(Pos_Captions)
                    simple_person = random.choice(simple_Captions3)

                context_simple = simple_person
                context = caption_person
                try:
                    captions = caption_image(
                        img, args, net, preprocess, context=context
                    )
                except Exception as e:
                    print(e)
                    print("Error with captioning")
                    embed2 = discord.Embed(
                        title=f"Caption error",
                        description=f"{e}",
                        color=discord.Color.red(),
                    )
                    await ctx.send(embed=embed2)
                for c in captions[: args.display]:
                    sendable = f"`{context}` {c}"
                    await ctx.reply(sendable, mention_author=False)
                    await ctx.add_reaction(first_line)
            except Exception as e:
                print(e)
                print("error")
                embed = discord.Embed(
                    title=f"Caption error",
                    description=f"{e}",
                    color=discord.Color.red(),
                )
                await ctx.send(embed=embed)
            torch.cuda.empty_cache()
    else:
        return


@bot.command()
async def imagine(ctx):
    if ctx.channel.id != channel_id:
        return
    gc.collect()
    torch.cuda.empty_cache()
    print("Command Loaded")
    async with ctx.channel.typing():
        if ctx.message.attachments:
            link = ctx.message.attachments[0].url
            filename = link.split("/")[-1]
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)
            image_prompts2 = f"{filename}"
        else:
            image_prompts2 = None
        width = os.environ["width"]
        width = int(width)
        height = os.environ["height"]
        height = int(height)
        img = Image.new("RGB", (width, height), color="black")
        img.save("progress.png")
        # definitions
        author = ctx.message.author.id
        input = ctx.message.content
        prompt = input[9 : len(input)]
        await bot.change_presence(activity=discord.Game(name=f"Imagining {prompt}"))

        if os.environ["seed"] != "42":
            seed = os.environ["seed"]
        if os.environ["seed"] == "0" or 0:
            seed = np.random.seed(0)
            seed = torch.seed()
            torch.manual_seed(seed)
        else:
            seed = np.random.seed(0)
            seed = torch.seed()
            torch.manual_seed(seed)
        os.environ["seed"] = str(seed)

        with open("averages.txt", "r") as f:
            orders = [name.rstrip() for name in f]
            orders = [float(item) for item in orders]

        average = sum(orders) / len(orders)
        average = str(round(average, 2))
        average = float(average)

        iterations = os.environ["max_iterations"]
        iterations = int(iterations)
        it_s = average
        epochs = 1

        num = iterations / it_s
        num = num * epochs
        num_elapsed = iterations / it_s

        secs = 1
        if num > 420:
            num = 7
            secs = num_elapsed - 420
        if num > 360:
            num = 6
            secs = num_elapsed - 360
        if num > 300:
            num = 5
            secs = num_elapsed - 300
        if num > 240:
            num = 4
            secs = num_elapsed - 240
        if num > 180:
            num = 3
            secs = num_elapsed - 180
        if num > 120:
            num = 2
            secs = num_elapsed - 120
        if num > 60:
            num = 1
            secs = num_elapsed - 60
        secs = int(secs)
        time.sleep(3)
        await ctx.channel.send(
            "```Imagining... ```"
            + f"**_{prompt}_**"
            + f"```Estimated: {num}m{secs}s Generation Time\nUsing seed: {seed}```"
        )
        message = await ctx.send(f"```    ```")

        secret_channel = bot.get_channel(
            955787994286129172
        )  # where 12345 would be your secret channel id
        file = discord.File(f"progress.png")
        temp_image = await ctx.send("```    ```")

        # image_message = await ctx.send(file = discord.File("progress.png"))
        genTime = time.time()
        os.environ["prompt"] = prompt

        try:
            # Check for GPU and reduce the default image size if low VRAM
            default_image_size = 512  # >8GB VRAM
            if not torch.cuda.is_available():
                default_image_size = 412  # no GPU found
            elif (
                get_device_properties(0).total_memory <= 2 ** 33
            ):  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
                default_image_size = 318  # <8GB VRAM

            width = os.environ["width"]
            width = int(width)
            height = os.environ["height"]
            height = int(height)
            max_iterations = os.environ["max_iterations"]
            max_iterations = int(max_iterations)
            os_seed = os.environ["seed"]
            seed = int(os_seed)
            prompt = os.environ["prompt"]
            model = os.environ["model"]

            def parse():
                global all_phrases
                vq_parser = argparse.ArgumentParser(
                    description="Image generation using VQGAN+CLIP"
                )

                vq_parser.add_argument(
                    "-aug",
                    "--augments",
                    nargs="+",
                    action="append",
                    type=str,
                    choices=["Hf", "Ji", "Sh", "Pe", "Ro", "Af", "Et", "Ts", "Er"],
                    help="Enabled augments (latest vut method only)",
                    default=[["Hf", "Af", "Pe", "Ji", "Er"]],
                    dest="augments",
                )
                vq_parser.add_argument(
                    "-cd",
                    "--cuda_device",
                    type=str,
                    help="Cuda device to use",
                    default="cuda:0",
                    dest="cuda_device",
                )
                vq_parser.add_argument(
                    "-ckpt",
                    "--vqgan_checkpoint",
                    type=str,
                    help="VQGAN checkpoint",
                    default=f"{model}.ckpt",
                    dest="vqgan_checkpoint",
                )
                vq_parser.add_argument(
                    "-conf",
                    "--vqgan_config",
                    type=str,
                    help="VQGAN config",
                    default=f"{model}.yaml",
                    dest="vqgan_config",
                )
                vq_parser.add_argument(
                    "-cpe",
                    "--change_prompt_every",
                    type=int,
                    help="Prompt change frequency",
                    default=0,
                    dest="prompt_frequency",
                )
                vq_parser.add_argument(
                    "-cutm",
                    "--cut_method",
                    type=str,
                    help="Cut method",
                    choices=["original", "latest"],
                    default="latest",
                    dest="cut_method",
                )
                vq_parser.add_argument(
                    "-cutp",
                    "--cut_power",
                    type=float,
                    help="Cut power",
                    default=1.0,
                    dest="cut_pow",
                )
                vq_parser.add_argument(
                    "-cuts",
                    "--num_cuts",
                    type=int,
                    help="Number of cuts",
                    default=32,
                    dest="cutn",
                )
                vq_parser.add_argument(
                    "-d",
                    "--deterministic",
                    action="store_true",
                    help="Enable cudnn.deterministic?",
                    dest="cudnn_determinism",
                )
                vq_parser.add_argument(
                    "-i",
                    "--iterations",
                    type=int,
                    help="Number of iterations",
                    default=max_iterations,
                    dest="max_iterations",
                )
                vq_parser.add_argument(
                    "-ifps",
                    "--input_video_fps",
                    type=float,
                    help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)",
                    default=15,
                    dest="input_video_fps",
                )
                vq_parser.add_argument(
                    "-ii",
                    "--init_image",
                    type=str,
                    help="Initial image",
                    default=None,
                    dest="init_image",
                )
                vq_parser.add_argument(
                    "-in",
                    "--init_noise",
                    type=str,
                    help="Initial noise image (pixels or gradient)",
                    default=None,
                    dest="init_noise",
                )
                vq_parser.add_argument(
                    "-ip",
                    "--image_prompts",
                    type=str,
                    help="Image prompts / target image",
                    default=[],
                    dest="image_prompts",
                )
                vq_parser.add_argument(
                    "-iw",
                    "--init_weight",
                    type=float,
                    help="Initial weight",
                    default=0.0,
                    dest="init_weight",
                )
                vq_parser.add_argument(
                    "-lr",
                    "--learning_rate",
                    type=float,
                    help="Learning rate",
                    default=0.1,
                    dest="step_size",
                )
                vq_parser.add_argument(
                    "-m",
                    "--clip_model",
                    type=str,
                    help="CLIP model (e.g. ViT-B/32, ViT-B/16)",
                    default="ViT-B/32",
                    dest="clip_model",
                )
                vq_parser.add_argument(
                    "-nps",
                    "--noise_prompt_seeds",
                    nargs="*",
                    type=int,
                    help="Noise prompt seeds",
                    default=[],
                    dest="noise_prompt_seeds",
                )
                vq_parser.add_argument(
                    "-npw",
                    "--noise_prompt_weights",
                    nargs="*",
                    type=float,
                    help="Noise prompt weights",
                    default=[],
                    dest="noise_prompt_weights",
                )
                vq_parser.add_argument(
                    "-o",
                    "--output",
                    type=str,
                    help="Output filename",
                    default="progress.png",
                    dest="output",
                )
                vq_parser.add_argument(
                    "-ofps",
                    "--output_video_fps",
                    type=float,
                    help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)",
                    default=0,
                    dest="output_video_fps",
                )
                vq_parser.add_argument(
                    "-opt",
                    "--optimiser",
                    type=str,
                    help="Optimiser",
                    choices=[
                        "Adam",
                        "AdamW",
                        "Adagrad",
                        "Adamax",
                        "DiffGrad",
                        "AdamP",
                        "RAdam",
                        "RMSprop",
                    ],
                    default="Adam",
                    dest="optimiser",
                )
                vq_parser.add_argument(
                    "-p",
                    "--prompts",
                    type=str,
                    help="Text prompts",
                    default=prompt,
                    dest="prompts",
                )
                vq_parser.add_argument(
                    "-s",
                    "--size",
                    nargs=2,
                    type=int,
                    help="Image size (width height) (default: %(default)s)",
                    default=[width, height],
                    dest="size",
                )
                vq_parser.add_argument(
                    "-sd", "--seed", type=int, help="Seed", default=seed, dest="seed"
                )
                vq_parser.add_argument(
                    "-se",
                    "--save_every",
                    type=int,
                    help="Save image iterations",
                    default=10,
                    dest="display_freq",
                )
                vq_parser.add_argument(
                    "-vid",
                    "--video",
                    action="store_true",
                    help="Create video frames?",
                    dest="make_video",
                )
                vq_parser.add_argument(
                    "-vl",
                    "--video_length",
                    type=float,
                    help="Video length in seconds (not interpolated)",
                    default=10,
                    dest="video_length",
                )
                vq_parser.add_argument(
                    "-vsd",
                    "--video_style_dir",
                    type=str,
                    help="Directory with video frames to style",
                    default=None,
                    dest="video_style_dir",
                )
                vq_parser.add_argument(
                    "-zs",
                    "--zoom_start",
                    type=int,
                    help="Zoom start iteration",
                    default=0,
                    dest="zoom_start",
                )
                vq_parser.add_argument(
                    "-zsc",
                    "--zoom_scale",
                    type=float,
                    help="Zoom scale %",
                    default=0.99,
                    dest="zoom_scale",
                )
                vq_parser.add_argument(
                    "-zse",
                    "--zoom_save_every",
                    type=int,
                    help="Save zoom image iterations",
                    default=10,
                    dest="zoom_frequency",
                )
                vq_parser.add_argument(
                    "-zsx",
                    "--zoom_shift_x",
                    type=int,
                    help="Zoom shift x (left/right) amount in pixels",
                    default=0,
                    dest="zoom_shift_x",
                )
                vq_parser.add_argument(
                    "-zsy",
                    "--zoom_shift_y",
                    type=int,
                    help="Zoom shift y (up/down) amount in pixels",
                    default=0,
                    dest="zoom_shift_y",
                )
                vq_parser.add_argument(
                    "-zvid",
                    "--zoom_video",
                    action="store_true",
                    help="Create zoom video?",
                    dest="make_zoom_video",
                )

                args = vq_parser.parse_args()
                if image_prompts2 is not None:
                    args.image_prompts = image_prompts2
                if not args.prompts and not args.image_prompts:
                    raise Exception("You must supply a text or image prompt")

                torch.backends.cudnn.deterministic = args.cudnn_determinism

                # Split text prompts using the pipe character (weights are split later)
                if args.prompts:
                    # For stories, there will be many phrases
                    story_phrases = [
                        phrase.strip() for phrase in args.prompts.split("^")
                    ]

                    # Make a list of all phrases
                    all_phrases = []
                    for phrase in story_phrases:
                        all_phrases.append(phrase.split("|"))
                        all_phrases.append(phrase.split(","))

                    # First phrase
                    args.prompts = all_phrases[0]

                # Split target images using the pipe character (weights are split later)
                if args.image_prompts:
                    args.image_prompts = args.image_prompts.split("|")
                    args.image_prompts = args.image_prompts.split(",")
                    args.image_prompts = [image.strip() for image in args.image_prompts]

                if args.make_video and args.make_zoom_video:
                    print(
                        "Warning: Make video and make zoom video are mutually exclusive."
                    )
                    args.make_video = False

                # Make video steps directory
                args.make_video = True
                if args.make_video or args.make_zoom_video:
                    if not os.path.exists("steps"):
                        os.mkdir("steps")

                return args

            class Prompt(nn.Module):
                def __init__(self, embed, weight=1.0, stop=float("-inf")):
                    super().__init__()
                    self.register_buffer("embed", embed)
                    self.register_buffer("weight", torch.as_tensor(weight))
                    self.register_buffer("stop", torch.as_tensor(stop))

                def forward(self, input):
                    input_normed = F.normalize(input.unsqueeze(1), dim=2)
                    embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
                    dists = (
                        input_normed.sub(embed_normed)
                        .norm(dim=2)
                        .div(2)
                        .arcsin()
                        .pow(2)
                        .mul(2)
                    )
                    dists = dists * self.weight.sign()
                    return (
                        self.weight.abs()
                        * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
                    )

            # NR: Split prompts and weights
            def split_prompt(prompt):
                vals = prompt.rsplit(":", 2)
                vals = vals + ["", "1", "-inf"][len(vals) :]
                return vals[0], float(vals[1]), float(vals[2])

            def load_vqgan_model(config_path, checkpoint_path):
                global gumbel
                gumbel = False
                config = OmegaConf.load(config_path)
                if config.model.target == "taming.models.vqgan.VQModel":
                    model = vqgan.VQModel(**config.model.params)
                    model.eval().requires_grad_(False)
                    model.init_from_ckpt(checkpoint_path)
                elif config.model.target == "taming.models.vqgan.GumbelVQ":
                    model = vqgan.GumbelVQ(**config.model.params)
                    model.eval().requires_grad_(False)
                    model.init_from_ckpt(checkpoint_path)
                    gumbel = True
                elif (
                    config.model.target
                    == "taming.models.cond_transformer.Net2NetTransformer"
                ):
                    parent_model = cond_transformer.Net2NetTransformer(
                        **config.model.params
                    )
                    parent_model.eval().requires_grad_(False)
                    parent_model.init_from_ckpt(checkpoint_path)
                    model = parent_model.first_stage_model
                else:
                    raise ValueError(f"unknown model type: {config.model.target}")
                del model.loss
                return model

            # Vector quantize
            def synth(z):
                if gumbel:
                    z_q = vector_quantize(
                        z.movedim(1, 3), model.quantize.embed.weight
                    ).movedim(3, 1)
                else:
                    z_q = vector_quantize(
                        z.movedim(1, 3), model.quantize.embedding.weight
                    ).movedim(3, 1)
                return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

            @torch.inference_mode()
            def checkin(i, losses):
                losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
                tqdm.write(
                    f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}"
                )
                out = synth(z)
                info = PngImagePlugin.PngInfo()
                info.add_text("comment", f"{args.prompts}")
                TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info)

            def ascend_txt(i):
                # global i
                out = synth(z)
                iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

                result = []

                if args.init_weight:
                    # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
                    result.append(
                        F.mse_loss(z, torch.zeros_like(z_orig))
                        * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight)
                        / 2
                    )

                for prompt in pMs:
                    result.append(prompt(iii))

                if args.make_video:
                    img = np.array(
                        out.mul(255)
                        .clamp(0, 255)[0]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(np.uint8)
                    )[:, :, :]
                    img = np.transpose(img, (1, 2, 0))
                    imageio.imwrite("./steps/" + str(i) + ".png", np.array(img))

                return result  # return loss

            def train(i):
                opt.zero_grad(set_to_none=True)
                lossAll = ascend_txt(i)

                if i % args.display_freq == 0:
                    checkin(i, lossAll)

                loss = sum(lossAll)
                loss.backward()
                opt.step()

                # with torch.no_grad():
                with torch.inference_mode():
                    z.copy_(z.maximum(z_min).minimum(z_max))

            if __name__ == "__main__":

                args = parse()

                # Do it
                device = torch.device(args.cuda_device)
                model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(
                    device
                )
                jit = (
                    True
                    if version.parse(torch.__version__) < version.parse("1.8.0")
                    else False
                )
                perceptor = (
                    clip.load(args.clip_model, jit=jit)[0]
                    .eval()
                    .requires_grad_(False)
                    .to(device)
                )

                cut_size = perceptor.visual.input_resolution
                f = 2 ** (model.decoder.num_resolutions - 1)

                # Cutout class options:
                # 'latest','original','updated' or 'updatedpooling'
                if args.cut_method == "latest":
                    make_cutouts = MakeCutouts(args, cut_size, args.cutn)
                elif args.cut_method == "original":
                    make_cutouts = MakeCutoutsOrig(args, cut_size, args.cutn)

                toksX, toksY = args.size[0] // f, args.size[1] // f
                sideX, sideY = toksX * f, toksY * f

                # Gumbel or not?
                if gumbel:
                    e_dim = 256
                    n_toks = model.quantize.n_embed
                    z_min = model.quantize.embed.weight.min(dim=0).values[
                        None, :, None, None
                    ]
                    z_max = model.quantize.embed.weight.max(dim=0).values[
                        None, :, None, None
                    ]
                else:
                    e_dim = model.quantize.e_dim
                    n_toks = model.quantize.n_e
                    z_min = model.quantize.embedding.weight.min(dim=0).values[
                        None, :, None, None
                    ]
                    z_max = model.quantize.embedding.weight.max(dim=0).values[
                        None, :, None, None
                    ]

                if args.init_image:
                    if "http" in args.init_image:
                        img = Image.open(urlopen(args.init_image))
                    else:
                        img = Image.open(args.init_image)
                    pil_image = img.convert("RGB")
                    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                    pil_tensor = TF.to_tensor(pil_image)
                    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                elif args.init_noise == "pixels":
                    img = random_noise_image(args.size[0], args.size[1])
                    pil_image = img.convert("RGB")
                    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                    pil_tensor = TF.to_tensor(pil_image)
                    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                elif args.init_noise == "gradient":
                    img = random_gradient_image(args.size[0], args.size[1])
                    pil_image = img.convert("RGB")
                    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                    pil_tensor = TF.to_tensor(pil_image)
                    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                else:
                    one_hot = F.one_hot(
                        torch.randint(n_toks, [toksY * toksX], device=device), n_toks
                    ).float()
                    # z = one_hot @ model.quantize.embedding.weight
                    if gumbel:
                        z = one_hot @ model.quantize.embed.weight
                    else:
                        z = one_hot @ model.quantize.embedding.weight

                    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
                    # z = torch.rand_like(z)*2						# NR: check

                z_orig = z.clone()
                z.requires_grad_(True)
                pMs = []
                normalize = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )

                # CLIP tokenize/encode
                if args.prompts:
                    for prompt in args.prompts:
                        txt, weight, stop = split_prompt(prompt)
                        embed = perceptor.encode_text(
                            clip.tokenize(txt).to(device)
                        ).float()
                        pMs.append(Prompt(embed, weight, stop).to(device))

                for prompt in args.image_prompts:
                    path, weight, stop = split_prompt(prompt)
                    img = Image.open(path)
                    pil_image = img.convert("RGB")
                    img = resize_image(pil_image, (sideX, sideY))
                    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                    embed = perceptor.encode_image(normalize(batch)).float()
                    pMs.append(Prompt(embed, weight, stop).to(device))

                for seed, weight in zip(
                    args.noise_prompt_seeds, args.noise_prompt_weights
                ):
                    gen = torch.Generator().manual_seed(seed)
                    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(
                        generator=gen
                    )
                    pMs.append(Prompt(embed, weight).to(device))

                # Set the optimiser
                opt, z = get_opt(args.optimiser, z, args.step_size)

                # Output for the user
                print("Using device:", device)
                print("Optimising using:", args.optimiser)

                if args.prompts:
                    print("Using text prompts:", args.prompts)
                if args.image_prompts:
                    print("Using image prompts:", args.image_prompts)
                if args.init_image:
                    print("Using initial image:", args.init_image)
                if args.noise_prompt_weights:
                    print("Noise prompt weights:", args.noise_prompt_weights)

                if args.seed is None:
                    seed = torch.seed()
                else:
                    seed = args.seed
                torch.manual_seed(seed)
                print("Using seed:", seed)

                i = 0  # Iteration counter
                j = 0  # Zoom video frame counter
                p = 1  # Phrase counter
                smoother = 0  # Smoother counter
                this_video_frame = 0  # for video styling
                start_time = dt.datetime.today().timestamp()

                with tqdm() as pbar:
                    while i < args.max_iterations:
                        # Change text prompt
                        if args.prompt_frequency > 0:
                            if i % args.prompt_frequency == 0 and i > 0:
                                # In case there aren't enough phrases, just loop
                                if p >= len(all_phrases):
                                    p = 0

                                pMs = []
                                args.prompts = all_phrases[p]

                                # Show user we're changing prompt
                                print(args.prompts)

                                for prompt in args.prompts:
                                    txt, weight, stop = split_prompt(prompt)
                                    embed = perceptor.encode_text(
                                        clip.tokenize(txt).to(device)
                                    ).float()
                                    pMs.append(Prompt(embed, weight, stop).to(device))
                                p += 1
                        percent = i / 400
                        percent = percent * 100
                        percent = int(percent)

                        if i % args.display_freq == 0:
                            temp_message = await secret_channel.send(
                                file=discord.File("progress.png")
                            )
                            attachment = temp_message.attachments[0]
                            # await image_message.edit(content=image_message)
                            await message.edit(content=f"```{pbar} ({percent}%)```")
                            await temp_image.edit(content=attachment.url)
                        train(i)
                        i += 1
                        pbar.update()

                print("done")
                await message.edit(content=f"``` done! ```")

            init_frame = 0  # This is the frame where the video will start
            last_frame = i  # You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

            min_fps = 10
            max_fps = 60

            total_frames = last_frame - init_frame

            length = 15  # Desired time of the video in seconds

            frames = []
            tqdm.write("Generating video...")
            for i in range(init_frame, last_frame):  #
                frames.append(Image.open("./steps/" + str(i) + ".png"))
                size = (width, height)

            savepath = "./"
            imageio.mimsave(os.path.join(savepath, "movie.mp4"), frames)
            end = time.time() - start
            end = end / 60
            print(end)

        except Exception as e:
            print(e)
            print("Error")
            embed = discord.Embed(
                title=f"Generation error",
                description=f"{e}",
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)
        print("Generation Complete")

        new_prompt = "progress.png"
        ittime = time.time() - genTime
        print(f"Generation Time: {ittime}")
        elapsed = time.time() - genTime
        math_elapsed = time.time() - genTime
        secs = 1
        if elapsed > 420:
            elapsed = 7
            secs = math_elapsed - 420
        if elapsed > 360:
            elapsed = 6
            secs = math_elapsed - 360
        if elapsed > 300:
            elapsed = 5
            secs = math_elapsed - 300
        if elapsed > 240:
            elapsed = 4
            secs = math_elapsed - 240
        if elapsed > 180:
            elapsed = 3
            secs = math_elapsed - 180
        if elapsed > 120:
            elapsed = 2
            secs = math_elapsed - 120
        if elapsed > 60:
            elapsed = 1
            secs = math_elapsed - 60
        print(f"Elapsed Time: {elapsed}")
        elapsed = str(elapsed)
        secs = int(secs)
        elapsed = f"{elapsed}m{secs}s"
        it = iterations / ittime
        it = "{:.2f}".format(it)
        it = str(it)
        in_loc = "movie.mp4"
        out_loc = "generated_out.mp4"
        clip2 = VideoFileClip(in_loc)
        print("fps: {}".format(clip2.fps))
        clip2 = clip2.set_fps(clip2.fps * 5)
        final = clip2.fx(vfx.speedx, 5)
        print("fps: {}".format(final.fps))
        final.write_videofile(out_loc)
        # await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))

        with open("averages.txt", "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")
            file_object.write(it)

        with open("averages.txt", "r") as f:
            orders = [name.rstrip() for name in f]
            orders = [float(item) for item in orders]

        torch.cuda.empty_cache()
        temp_message2 = await secret_channel.send(file=discord.File(out_loc))
        attachment2 = temp_message2.attachments[0]
        upscale("progress.png", "progress.png")
        temp_message3 = await secret_channel.send(file=discord.File("progress.png"))
        attachment3 = temp_message3.attachments[0]
        await temp_image.edit(content=f"{attachment3.url}\n{attachment2.url}")

        # delete output
        directory = os.getcwd()
        my_dir = directory
        for fname in os.listdir(my_dir):
            if fname.endswith(".mp4"):
                os.remove(os.path.join(my_dir, fname))


@bot.command()
async def faces(ctx):
    await bot.change_presence(activity=discord.Game(name=f"Finding Faces"))
    if ctx.message.attachments:
        if ctx.channel.id != channel_id:
            return
        async with ctx.channel.typing():
            link = ctx.message.attachments[0].url
            filename = "urmom.png"
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)

            for face_location in face_locations:
                top, right, bottom, left = face_location

                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save("face.png")
                image = Image.open("face.png")
                new_image = image.resize((256, 256))
                new_image.save("face.png")
            try:
                test_image = face_recognition.load_image_file("face.png")
                face_locations = face_recognition.face_locations(test_image)
                face_encodings = face_recognition.face_encodings(
                    test_image, face_locations
                )
            except FileNotFoundError as e:
                embed = discord.Embed(
                    title=f"{ctx.command} error",
                    description=f"Error: Face Not Found",
                    color=discord.Color.red(),
                )
                await ctx.send(embed=embed)

            pil_image = Image.fromarray(test_image)
            draw = ImageDraw.Draw(pil_image)

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )

                name = "Unknown Person"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
                text_width, text_height = draw.textsize(name)
                draw.rectangle(
                    ((left, bottom - text_height - 10), (right, bottom)),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
                draw.text(
                    (left + 6, bottom - text_height - 5), name, fill=(255, 255, 255)
                )

            del draw

            pil_image.save("identify.png")
            await ctx.reply(file=discord.File("identify.png"), mention_author=False)
            # delete output
            directory = os.getcwd()
            my_dir = directory
            for fname in os.listdir(my_dir):
                if fname.endswith(".png"):
                    os.remove(os.path.join(my_dir, fname))
                if fname.endswith(".jpg"):
                    os.remove(os.path.join(my_dir, fname))
            await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))


@bot.command()
async def facehq(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Model Updating...")
        await ctx.channel.send("Updated! YAKbot Model: `vqgan_faceshq`")
        os.environ["model"] = "vqgan_faceshq"
        os.environ["height"] = "412"
        os.environ["width"] = "412"
        os.environ["max_iterations"] = "125"


@bot.command()
async def wikiart(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Model Updating...")
        await ctx.channel.send("Updated! YAKbot Model: `wikiart_16384`")
        os.environ["model"] = "wikiart_16384"
        os.environ["height"] = "288"
        os.environ["width"] = "512"
        os.environ["max_iterations"] = "400"


@bot.command()
async def default(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Model Updating...")
        await ctx.channel.send("Updated! YAKbot Model: `vqgan_imagenet_f16_16384`")
        os.environ["model"] = "vqgan_imagenet_f16_16384"
        os.environ["height"] = "412"
        os.environ["width"] = "412"
        os.environ["max_iterations"] = "400"


@bot.command()
async def d1024(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Model Updating...")
        await ctx.channel.send("Updated! YAKbot Model: `vqgan_imagenet_f16_16384`")
        os.environ["model"] = "vqgan_imagenet_f16_1024"
        os.environ["height"] = "412"
        os.environ["width"] = "412"
        os.environ["max_iterations"] = "400"


@bot.command()
async def esrgan(ctx):
    await bot.change_presence(activity=discord.Game(name=f"Enlarging Images"))
    torch.cuda.empty_cache()
    if ctx.message.attachments:
        if ctx.channel.id != channel_id:
            return
        async with ctx.channel.typing():
            link = ctx.message.attachments[0].url
            filename = "input.png"
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)
            upscale(filename, "progress_out.png")
            try:
                output_path = "progress_out.png"
                await ctx.reply(file=discord.File(output_path), mention_author=False)
            except FileNotFoundError as error:
                print(error)
                await ctx.channel.send(
                    "`Error: Image to large to be upscaled. Please try a smaller image.`"
                )
            # delete output
            directory = os.getcwd()
            my_dir = directory
            for fname in os.listdir(my_dir):
                if fname.endswith(".png"):
                    os.remove(os.path.join(my_dir, fname))
                if fname.endswith(".jpg"):
                    os.remove(os.path.join(my_dir, fname))
            torch.cuda.empty_cache()
            await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))


@bot.command()
async def rembg(ctx):
    await bot.change_presence(activity=discord.Game(name=f"Removing Background"))
    if ctx.message.attachments:
        if ctx.channel.id != channel_id:
            return
        async with ctx.channel.typing():
            link = ctx.message.attachments[0].url
            filename = link.split("/")[-1]
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)
            input_path = filename
            output_path = "out.png"

            f = np.fromfile(input_path)
            result = remove(f)
            img = Image.open(io.BytesIO(result)).convert("RGBA")
            img.save(output_path)
            await ctx.reply(file=discord.File(output_path), mention_author=False)

            # delete output
            directory = os.getcwd()
            my_dir = directory
            for fname in os.listdir(my_dir):
                if fname.endswith(".png"):
                    os.remove(os.path.join(my_dir, fname))
            torch.cuda.empty_cache()
            await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))


@bot.command()
async def help(ctx):
    await bot.change_presence(activity=discord.Game(name=f"Helping"))
    if ctx.channel.id != channel_id:
        return
    print("Command Loaded")
    async with ctx.channel.typing():
        embed = discord.Embed(
            title="YAKbot Help",
            description=f"`.rembg [Attached Image]`\n**removes background from attatched image**\n\n`.esrgan [Attatchment]`\n**YAKbot will use a pretrained ESRGAN upscaler to upscale you images resolution by up to 4 times**\n\n`.status`\n**sends embed message with all relevent device stats for YAKbot**\n\n`.imagine [Prompt]`\n**uses CLIP+VQGAN open generation to create an original image from your prompt**\n\n`.facehq, .wikiart, .default, .d1024`\n**Changes YAKbots VQGAN+CLIP model to one trained solely on faces, art or default configuration**\n\n`.square, .landscape, .portrait`\n**YAKbot will update his size configurations for generations to your specified orientation**\n\n`.seed [Desired Seed]`\n**Changes YAKbots seed for all open generation (if 0 will set to random)**\n\n`.faces [Attatchment]`\n**YAKbot will look through your photo and try to find any recognizable faces**\n\n`.colorize [Attatchment]`\n**YAKbot will turn your black and white attatchment into a colorized version**\n\n`.outline [Prompt]`\n**YAKbot will contact a local GPT3 model that will synthasize and look for essays on your prompt while outputting an outline/list of ideas/facts about your prompt to help kickstart your projects**\n\n__Any Attatchments Sent In This Channel Will Be Identified And Captioned By YAKbot (To Prevent Captioning Include --nc In Your Message)__",
            color=MAIN_COLOR,
        )
        await ctx.channel.send(embed=embed)
    torch.cuda.empty_cache()
    await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))


@bot.command()
async def status(ctx):
    print("Command Loaded")
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        name = torch.cuda.get_device_name(0)
        available = torch.cuda.device_count()
        Used = "(Available)"
        if available == 1:
            available = (
                f"There is currently `{available}` device available for generation"
            )
            Used = "(Available)"

        if available == 2:
            available = (
                f"There are currently `{available}` devices available for generation"
            )

        if available == 0:
            available = (
                f"There are currently `{available}` devices available for generation"
            )
            Used = "(In Use)"
        status = f'Using device: {device}\n{name} `{Used}`\n\n{available}\n\nYAKbot is currently running on `{os.environ["model"]}`'

        embed = discord.Embed(
            title="Device Status", description=status, color=0x7289DA
        )  # ,color=Hex code

        print(status)
        await ctx.channel.send(embed=embed)
        torch.cuda.empty_cache()


@bot.command()
async def kill(ctx):
    if ctx.channel.id != channel_id:
        return
    print("Command Loaded")
    async with ctx.channel.typing():
        await ctx.channel.send("```Exiting```")
        await bot.logout()
        sys.exit()


@bot.command()
async def seed(ctx):
    if ctx.channel.id != channel_id:
        return
    print("Command Loaded")
    async with ctx.channel.typing():
        input = ctx.message.content
        seed = input[6 : len(input)]
        os.environ["seed"] = seed
        await ctx.channel.send(f"```Seed: {seed}```")
    torch.cuda.empty_cache()


@bot.command()
async def square(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Size Updating...")
        await ctx.channel.send("Updated! YAKbot Size Config: `square`")
        os.environ["height"] = "412"
        os.environ["width"] = "412"


@bot.command()
async def portrait(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Size Updating...")
        await ctx.channel.send("Updated! YAKbot Size Config: `portrait`")
        os.environ["height"] = "512"
        os.environ["width"] = "288"


@bot.command()
async def landscape(ctx):
    if ctx.channel.id != channel_id:
        return
    async with ctx.channel.typing():
        await ctx.channel.send("YAKbot Size Updating...")
        await ctx.channel.send("Updated! YAKbot Size Config: `landscape`")
        os.environ["height"] = "288"
        os.environ["width"] = "512"


@bot.command()
async def colorize(ctx):
    await bot.change_presence(activity=discord.Game(name=f"Colorizing..."))
    if ctx.message.attachments:
        if ctx.channel.id != channel_id:
            return
        async with ctx.channel.typing():
            link = ctx.message.attachments[0].url
            filename = "teaser.jpg"
            r = requests.get(link, allow_redirects=True)
            open(filename, "wb").write(r.content)
            print(filename)
            img_path = filename
            # default size to process images is 256x256
            # grab L channel in both original ("orig") and resized ("rs") resolutions
            img = load_img(img_path)
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
            if use_gpu:
                tens_l_rs = tens_l_rs.cuda()

            img_bw = postprocess_tens(
                tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
            )
            out_img_eccv16 = postprocess_tens(
                tens_l_orig, colorizer_eccv16(tens_l_rs).cpu()
            )
            out_img_siggraph17 = postprocess_tens(
                tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu()
            )

            plt.imsave("%s_eccv16.png" % save_prefix, out_img_eccv16)
            plt.imsave("%s_siggraph17.png" % save_prefix, out_img_siggraph17)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(img_bw)
            plt.title("Input")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(out_img_eccv16)
            plt.title("Output (ECCV 16)")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(out_img_siggraph17)
            plt.title("Output (SIGGRAPH 17)")
            plt.axis("off")
            plt.savefig("out.png")
            await ctx.reply(file=discord.File("out.png"), mention_author=False)
            # delete output
            directory = os.getcwd()
            my_dir = directory
            for fname in os.listdir(my_dir):
                if fname.endswith(".png"):
                    os.remove(os.path.join(my_dir, fname))
            torch.cuda.empty_cache()
            await bot.change_presence(activity=discord.Game(name=f"in a trash bin"))


@bot.command()
async def outline(ctx):
    if ctx.channel.id != channel_id:
        return
    print("Command Loaded")
    async with ctx.channel.typing():
        input = ctx.message.content
        question = str(input[9 : len(input)])

        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=f"Create an outline for an essay about {question}:",
            temperature=0,
            max_tokens=364,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        print(response.choices[0].text)
        embed = discord.Embed(
            title=f"Research outline for {question}",
            description=response.choices[0].text,
            color=0x7289DA,
        )
        await ctx.channel.send(embed=embed)


@bot.event
async def on_command_error(ctx, error):
    # if command has local error handler, return
    if hasattr(ctx.command, "on_error"):
        return

    # get the original exception
    error = getattr(error, "original", error)

    if isinstance(error, commands.CommandNotFound):
        return

    if isinstance(error, commands.BotMissingPermissions):
        missing = [
            perm.replace("_", " ").replace("guild", "server").title()
            for perm in error.missing_perms
        ]
        if len(missing) > 2:
            fmt = "{}, and {}".format("**, **".join(missing[:-1]), missing[-1])
        else:
            fmt = " and ".join(missing)
        _message = "I need the **{}** permission(s) to run this command.".format(fmt)
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description="I need the **{}** permission(s) to run this command.".format(
                fmt
            ),
            color=discord.Color.red(),
        )
        embed.set_footer(text=f"{error}")
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.DisabledCommand):
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description="This command has been disabled",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.CommandOnCooldown):
        remaining = "{}".format(str(datetime.timedelta(seconds=error.retry_after)))
        embed = discord.Embed(
            description=f"This command is on cooldown, please try again in "
            f"{remaining[0:1]} hours, "
            f"{remaining[3:4]} minutes, "
            f"{remaining[6:7]} seconds!\n"
            f"To avoid getting these cooldowns please vote by clicking above! This will "
            f"kick in within 1 minute and 30 seconds!",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    if isinstance(error, discord.HTTPException):
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description=f"{error.text}",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.MissingPermissions):
        missing = [
            perm.replace("_", " ").replace("guild", "server").title()
            for perm in error.missing_perms
        ]
        if len(missing) > 2:
            fmt = "{}, and {}".format("**, **".join(missing[:-1]), missing[-1])
        else:
            fmt = " and ".join(missing)
        _message = "You need the **{}** permission(s) to use this command.".format(fmt)
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description=f"{_message}",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    # This error is the most common and will need tweaking to how you setup your help command.
    if isinstance(error, commands.UserInputError):
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description=f"Invalid user input. "
            f"Please use `{bot.command_prefix}help {ctx.command.cog_name}` "
            f"and locate the `{ctx.command}` command. Check what arguments are "
            f"needed underneath it and retry this command!",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.NoPrivateMessage):
        try:
            embed = discord.Embed(
                title=f"{ctx.command} error",
                description="This command cannot be sued in direct messages",
                color=discord.Color.red(),
            )
            await ctx.author.send(embed=embed)
        except discord.Forbidden:
            pass
        return

    if isinstance(error, commands.CheckFailure):
        embed = discord.Embed(
            title=f"{ctx.command} error",
            description=f"You do not have permission to use this command",
            color=discord.Color.red(),
        )
        await ctx.send(embed=embed)
        return

    # ignore all other exception types, but print them to stderr
    print("Ignoring exception in command {}:".format(ctx.command), file=sys.stderr)

    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)


@bot.event
async def on_command(ctx):
    api.command_run(ctx)


bot.run(os.environ["bot_token"])
