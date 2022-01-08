import time
from clip.clip import available_models
startTime = time.time()
import argparse
import presets
import sys
from upscaler import upscale
from GPTJ.Basic_api import SimpleCompletion
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
os.chdir('C:\\Users\\noahs\\Desktop\\BATbot\\Bot')
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
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid

import model
import utils

import numpy as np
import faiss
import requests
import torchvision.transforms.functional as F

from PIL import Image
from torchvision.utils import make_grid

import model
import retrofit

from rembg.bg import remove
import numpy as np
import io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model2, preprocess2 = clip.load("ViT-B/32", device=device)
text2 = clip.tokenize(["negative", "neutral", "positive"]).to(device)
print('Using device:', device)
print("loading models")
load_categories = "emojis"
load(load_categories)

lowerBoundNote = 21
resolution = 0.25
MAIN_COLOR = 0x459fff  # light blue kinda

# filepaths
fp_in = "C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\*.png"
fp_out = "C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\movie.mp4"

#face recognition settings
#Noah
image_of_noah = face_recognition.load_image_file('./img/known/Noah.png')
noah_face_encoding = face_recognition.face_encodings(image_of_noah)[0]
#Jack
image_of_jack = face_recognition.load_image_file('./img/known/Jack.png')
jack_face_encoding = face_recognition.face_encodings(image_of_jack)[0]
#Ewan
image_of_ewan = face_recognition.load_image_file('./img/known/Ewan.png')
ewan_face_encoding = face_recognition.face_encodings(image_of_ewan)[0]
#Erin
image_of_erin = face_recognition.load_image_file('./img/known/Erin.png')
erin_face_encoding = face_recognition.face_encodings(image_of_erin)[0]
#Brent
image_of_brent = face_recognition.load_image_file('./img/known/Brent.png')
brent_face_encoding = face_recognition.face_encodings(image_of_brent)[0]
#Connor
image_of_connor = face_recognition.load_image_file('./img/known/Connor.png')
connor_face_encoding = face_recognition.face_encodings(image_of_connor)[0]
#Liam
image_of_liam = face_recognition.load_image_file('./img/known/Liam.png')
liam_face_encoding = face_recognition.face_encodings(image_of_liam)[0]
#Alek
image_of_alek = face_recognition.load_image_file('./img/known/Alek.png')
alek_face_encoding = face_recognition.face_encodings(image_of_alek)[0]

#  Create arrays of encodings and names
known_face_encodings = [
  noah_face_encoding,
  jack_face_encoding,
  ewan_face_encoding,
  erin_face_encoding,
  brent_face_encoding,
  connor_face_encoding,
  liam_face_encoding,
  alek_face_encoding
]

known_face_names = [
  "Noah",
  "Jack",
  "Ewan",
  "Erin",
  "Brent",
  "Connor/Ari",
  "Liam",
  "Alek"
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
        clip.tokenize(candidates).to(args.device)).float()
    textemb /= textemb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * x @ textemb.T).softmax(dim=-1)
    _, indices = similarity[0].topk(args.num_return_sequences)
    return [candidates[idx] for idx in indices[0]]

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def caption_image(path, args, net, preprocess, context):
    captions = []
    img, mat = load_image(path, preprocess)
    table, x = net.build_table(mat.half(),
                               net.perceiver,
                               ctx=context,
                               indices=net.indices,
                               indices_data=net.indices_data,
                               knn=args.knn,
                               tokenize=clip.tokenize,
                               device=args.device,
                               is_image=True,
                               return_images=True)
    
    table = net.tokenizer.encode(table[0], return_tensors='pt').to(device)
    table = table.squeeze()[:-1].unsqueeze(0)
    out = net.model.generate(table,
                             max_length=args.maxlen,
                             do_sample=args.do_sample,
                             num_beams=args.num_beams,
                             temperature=args.temperature,
                             top_p=args.top_p,
                             num_return_sequences=args.num_return_sequences)
    candidates = []
    for seq in out:
        decoded = net.tokenizer.decode(seq, skip_special_tokens=True)
        decoded = decoded.split('|||')[1:][0].strip()
        candidates.append(decoded)
    captions = clip_rescoring(args, net, candidates, x[None,:])
    print(f'Personality: {context[0]}\n')
    for c in captions[:args.display]:
        print(c)
    display_grid([img])
    return captions

def success_embed(title, description):
    return Embed(
        title=title,
        description=description,
        color=MAIN_COLOR
    )

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

async def play_source(voice_client):
    source = FFmpegPCMAudio("C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\audio\\krusty_krab_theme.mp3")
    voice_client.play(source, after=lambda e: print('Player error: %s' % e) if e else bot.loop.create_task(play_source(voice_client)))

# Settings
args = argparse.Namespace(
    config='./checkpoints/12xdqrwd-config',
    index_dirs='./unigrams,./bigrams,./artstyles,./emotions',
    clip_model='ViT-B/16',
    knn=3,
    maxlen=72,
    num_return_sequences=32,     # decrease this is you get GPU OOM, increase if using float16 model
    num_beams=1,
    temperature=0.8,
    top_p=0.9,
    display=1,
    do_sample=True,
    device=device
)

# Load indices
indices = []
indices_data = []
index_dirs = args.index_dirs.split(',')
index_dirs = list(filter(lambda t: len(t) > 0, index_dirs))
for index_dir in index_dirs:
    fname = os.path.join(index_dir, 'args.txt')
    with open(fname, 'r') as f:
        index_args = dotdict(json.load(f))

    entries = []
    fname = os.path.join(index_dir, 'entries.txt')
    with open(fname, 'r') as f:
        entries.extend([line.strip() for line in f])

    indices_data.append(entries)
    indices.append(faiss.read_index(glob.glob(f"{index_dir}/*.index")[0]))
preprocess = clip.load(args.clip_model, jit=False)[1]

# Load model
config = dotdict(torch.load(args.config))
config.task = 'txt2txt'
config.adapter = './checkpoints/12xdqrwd.ckpt'
net = retrofit.load_params(config).to(device)
net.indices = indices
net.indices_data = indices_data
print('loaded')

name = ["dodo82", "erintotoro", "jack-cheng", "alekk-stroms", "kabomton", "ari", "dentist sam", "sam"]
randomstatus = random.choice(name)
#Bot
bot = commands.Bot(command_prefix='.', help_command=None)

key = "statcord.com-pYzEX1uxAJv7PAT4ZaAx"
api = statcord.Client(bot,key)
api.start_loop()
channel_id = 882342184924348478
public_channel_id = 920889454443524116

@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))
    print('We have logged in as {0.user}'.format(bot))
    BATbot = ("""██████╗░░█████╗░████████╗██████╗░░█████╗░████████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝
██████╦╝███████║░░░██║░░░██████╦╝██║░░██║░░░██║░░░
██╔══██╗██╔══██║░░░██║░░░██╔══██╗██║░░██║░░░██║░░░
██████╦╝██║░░██║░░░██║░░░██████╦╝╚█████╔╝░░░██║░░░
╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═════╝░░╚════╝░░░░╚═╝░░░""")
    print(BATbot)
    print(f"Logged in as {bot.user}")
    print(f"Connected to: {len(bot.guilds)} guilds")
    print(f"Connected to: {len(bot.users)} users")
    print(f"Connected to: {len(bot.cogs)} cogs")
    print(f"Connected to: {len(bot.commands)} commands")
    print(f"Connected to: {len(bot.emojis)} emojis")
    print(f"Connected to: {len(bot.voice_clients)} voice clients")
    print(f"Connected to: {len(bot.private_channels)} private_channels")
    print(f'Loaded cogs: {bot.cogs}')
    print('Elapsed Startup Time: ', time.time() - startTime)

@bot.listen('on_message')
async def image(ctx):
 if ctx.author.bot:
  return
 if ctx.attachments and ctx.content != '.rembg' and ctx.content != '.sop' and ctx.content != '.faces' and ctx.content != '.esrgan' and ctx.content != '.img2sound':
    if ctx.channel.id != channel_id:
      return
    gc.collect()
    torch.cuda.empty_cache()
    async with ctx.channel.typing():
     await asyncio.sleep(0.1)
     #Download Image
     link = ctx.attachments[0].url
     filename = link.split('/')[-1]
     r = requests.get(link, allow_redirects=True)
     open(filename, 'wb').write(r.content)
     print(filename)

     #Caption+Emoji
     file_ = filename
     print("classifying")
     image2 = preprocess2(Image.open(file_)).unsqueeze(0).to(device)
     reaction = classify(file_)
     reaction = str(reaction)
     print(reaction)
     emoji = f"{reaction}"
     emoji = emoji.encode('unicode-escape').decode('ASCII')
     with open("reactables.txt", "a+") as file_object:
        file_object.write(emoji)
     with open('reactables.txt') as f:
        first_line = f.readline()
     os.remove('reactables.txt')
     first_line = first_line[:-2]
     first_line = first_line.encode('ascii').decode('unicode-escape').encode('utf-16', 'surrogatepass').decode('utf-16')
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

     Neg_Captions = ['dread', 'hurt', 'suffering', 'afraid', 'aggressive', 'alarmed', 'annoyed', 'anxious', 'bitter', 'brooding', 'claustrophobic', 'cowardly', 'demoralized', 'depressed', 'disheartened', 'disoriented', 'dispirited', 'disturbed', 'frustrated', 'gloomy', 'helpless', 'hopeless', 'horrified', 'impatient', 'indignant', 'infuriated', 'irritated', 'lonely', 'mad', 'miserable', 'mortified', 'nasty', 'nauseated', 'negative', 'nervous', 'offended', 'panicked', 'paranoid', 'possessive', 'powerless', 'rash', 'rejected', 'scared', 'shocked', 'smug', 'stressed', 'stubborn', 'stuck', 'suspicious', 'troubled', 'unhappy', 'unsettled', 'unsure', 'upset', 'vengeful', 'vicious', 'vulnerable', 'weak', 'worried', 'bad', 'bitter', 'cold', 'crazydead', 'dry', 'fat', 'hollow', 'old', 'plain', 'poor', 'shy', 'sore', 'sour', 'wrong', 'dark', 'shadow', 'anger', 'anxiousness', 'cruelty', 'cynic', 'denial', 'depression', 'despair', 'disgust', 'emptiness', 'fear', 'gray', 'grief', 'hate', 'hostile', 'hurt', 'indignation', 'insanity', 'irritability', 'jealousy', 'longing', 'lust', 'mourning', 'needy', 'pain', 'paranoia', 'pity', 'possessive', 'pride', 'rage', 'remorse', 'resentment', 'resignation', 'sadness', 'scorn', 'sensitive', 'shame', 'sorrow', 'tense', 'uncertainty', 'uneasiness', 'upset', 'agitation', 'agony', 'alarm', 'alienation', 'anguish', 'apathy', 'aversion', 'brooding', 'confusion', 'cynicism', 'dejection', 'disappointment', 'disbelief', 'discomfort', 'discontentment', 'displeasure', 'distraction', 'distress', 'doubt', 'dread', 'embarrassment', 'expectancy', 'fright', 'fury', 'glumness', 'greed', 'grumpiness', 'guilt', 'hatred', 'homesickness', 'humiliation', 'insecurity', 'loathing', 'miserliness', 'negative', 'neglect', 'outrage', 'paranoid', 'pessimism', 'rash', 'regret', 'restlessness', 'self-pity', 'spite', 'suffering', 'sullenness', 'suspense', 'tension', 'terror', 'woe', 'wrath']
     Neu_Captions = ['arrogant', 'assertive', 'astonished', 'baffled', 'bewildered', 'bored', 'brazen', 'cheeky', 'coercive', 'content', 'dazed', 'determined', 'discombobulated', 'disgruntled', 'dominant', 'dumbstruck', 'exasperated', 'flakey', 'hospitable', 'insightful', 'isolated', 'lazy', 'loopy', 'moody', 'mystified', 'numb', 'obstinate', 'perplexed', 'persevering', 'pleased', 'puzzled', 'rattled', 'reluctant', 'ruthless', 'self-conscious', 'submissive', 'tired', 'unnerved', 'alert', 'beige', 'clear', 'elderly', 'few', 'giant', 'granite', 'petite', 'quiet', 'shallow', 'sharp', 'solid', 'square', 'swift', 'wet', 'wild', 'pale', 'flamboyant', 'neutral', 'animosity', 'anticipation', 'camaraderie', 'cautious', 'content', 'poem', 'down', 'envy', 'expectation', 'infatuation', 'interest', 'irritability', 'longing', 'lust', 'mean', 'mercy', 'mildness', 'perturbation', 'pity', 'resignation', 'sensitive', 'tense', 'uncertainty', 'uneasiness', 'yearning', 'ambivalence', 'apathy', 'apprehension', 'attentiveness', 'aversion', 'baffled', 'confusion', 'curiosity', 'dismay', 'distraction', 'dominant', 'epiphany', 'expectancy', 'fascination', 'hope', 'humility', 'hysteria', 'idleness', 'indifference', 'jubilation', 'melancholy', 'modesty', 'patience', 'restlessness', 'revulsion', 'self-pity', 'sentimentality', 'serenity', 'suspense', 'tension']
     Pos_Captions = ['Ridiculous','caring', 'melancholy', 'calm', 'carefree', 'careless', 'comfortable', 'confident', 'delighted', 'driven', 'enchanted', 'enlightened', 'focused', 'kind', 'nostalgic', 'optimistic', 'positive', 'relaxed', 'relieved', 'self-confident', 'self-respecting', 'shameless', 'thrilled', 'triumphant', 'worthy', 'better', 'cool', 'fancy', 'good', 'light', 'modern', 'rich', 'safe', 'superior', 'sweet', 'tan', 'whispering', 'wise', 'nice', 'young', 'delicious', 'sweet', 'bold', 'sunny', 'flaming', 'gay', 'sparkling', 'shining', 'glitter', 'glowing', 'well', 'showing', 'bright', 'acceptance', 'adoration', 'affection', 'attraction', 'intellectual', 'spiritual', 'bliss', 'bubbly', 'calm', 'contempt', 'desire', 'eager', 'enlightened', 'exuberance', 'fulfillment', 'hopeful', 'peace', 'innocent', 'interest', 'joy', 'kind', 'longing', 'love', 'lust', 'melancholic', 'mercy', 'passion', 'pleasure', 'pride', 'relief', 'sensitive', 'sincerity', 'trust', 'yearning', 'admiration', 'amazement', 'ambivalence', 'amusement', 'attentiveness', 'baffled', 'sweetness', 'awe', 'caring', 'charity', 'cheerfulness', 'courage', 'curiosity', 'eagerness', 'ecstasy', 'empathy', 'enjoyment', 'enthusiasm', 'epiphany', 'euphoria', 'excitement', 'fascination', 'fondness', 'friendliness', 'glee', 'gratitude', 'happiness', 'hope', 'humility', 'liking', 'melancholy', 'modesty', 'patience', 'politeness', 'positive', 'satisfaction', 'sentimentality', 'surprise', 'sympathy', 'tenderness', 'thankfulness', 'tolerance', 'worthy']
     
     simple_Captions1 = ['negative']
     simple_Captions2 = ['neutral']
     simple_Captions3 = ['positive']

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

     context = simple_person
     captions = caption_image(img, args, net, preprocess, context=context)
     for c in captions[:args.display]:
      sendable = f'`{context}` {c}'
      await ctx.reply(sendable, mention_author=False)
      await ctx.add_reaction(first_line)
     torch.cuda.empty_cache()
 else:
   return

@bot.command()
async def imagine(ctx):
    if ctx.channel.id != channel_id:
      return
    gc.collect()
    torch.cuda.empty_cache()
    print('Command Loaded')
    print('Cache Prepared')
    async with ctx.channel.typing():
      #definitions
      author = ctx.message.author.id
      input = ctx.message.content
      prompt = input[9:len(input)]
      await bot.change_presence(activity=discord.Game(name=f"Imagining {prompt}"))
      
      if os.environ['seed'] != '42':
       seed = os.environ['seed']
      if os.environ['seed'] == '0' or 0:
       seed = np.random.seed(0)
       seed = torch.seed()
       torch.manual_seed(seed)
      else:
       seed = np.random.seed(0)
       seed = torch.seed()
       torch.manual_seed(seed)
      print(seed)
      os.environ['seed'] = str(seed)
      print(seed)

      with open ('averages.txt','r') as f :
        orders = [name.rstrip() for name in f]
        orders = [float(item) for item in orders]
        print(orders)
    
      average = sum(orders) / len(orders)
      average = str(round(average, 2))
      average = float(average)

      iterations = os.environ['max_iterations']
      iterations = int(iterations)
      it_s = average
      epochs=1

      num = iterations/it_s
      num = num*epochs
      num_elapsed = iterations/it_s

      secs = 1
      if num > 420:
        num = 7
        secs = num_elapsed-420
      if num > 360:
        num = 6
        secs = num_elapsed-360
      if num > 300:
        num = 5 
        secs = num_elapsed-300
      if num > 240:
        num = 4
        secs = num_elapsed-240
      if num > 180:
        num = 3 
        secs = num_elapsed-180
      if num > 120:
        num = 2
        secs = num_elapsed-120
      if num > 60:
        num = 1
        secs = num_elapsed-60
      secs = int(secs)
      time.sleep(3)
      await ctx.channel.send("```Imagining... ```"+f"**_{prompt}_**"+f'```Estimated: {num}m{secs}s Generation Time\nUsing seed: {seed}```')
      genTime = time.time()

      print(os.environ['prompt']) # outputs 'value'
      os.environ['prompt'] = prompt
      print(os.environ['prompt']) # outputs 'newvalue'

      subprocess.call(f'python3 VQGAN_CLIP.py')
      print('Generation Complete')
      
      new_prompt = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\progress.png'
      upscale(new_prompt, 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\progress.png')
      await ctx.channel.send(f'<@{author}>'+'```Your Generation Is Done```')
      await ctx.channel.send(file=discord.File(new_prompt))
      ittime = time.time() - genTime
      print(f'Generation Time: {ittime}')
      elapsed = time.time() - genTime
      math_elapsed = time.time() - genTime
      secs = 1
      if elapsed > 420:
        elapsed = 7 
        secs = math_elapsed-420
      if elapsed > 360:
        elapsed = 6 
        secs = math_elapsed-360
      if elapsed > 300:
        elapsed = 5 
        secs = math_elapsed-300
      if elapsed > 240:
        elapsed = 4
        secs = math_elapsed-240
      if elapsed > 180:
        elapsed = 3 
        secs = math_elapsed-180
      if elapsed > 120:
        elapsed = 2
        secs = math_elapsed-120
      if elapsed > 60:
        elapsed = 1
        secs = math_elapsed-60
      print(f'Elapsed Time: {elapsed}')
      elapsed = str(elapsed)
      secs = int(secs)
      elapsed = f'{elapsed}m{secs}s'
      it = iterations / ittime
      it = "{:.2f}".format(it)
      it = str(it)
      in_loc = 'movie.mp4'
      out_loc = 'generated_out.mp4'
      clip = VideoFileClip(in_loc)
      print("fps: {}".format(clip.fps))
      clip = clip.set_fps(clip.fps * 5)
      final = clip.fx(vfx.speedx, 5)
      print("fps: {}".format(final.fps))
      final.write_videofile(out_loc)
      await ctx.channel.send(file=discord.File(out_loc))
      await ctx.channel.send(f'```Elapsed: {elapsed} | Generated at: {it}it/s```')
      await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

      with open("averages.txt", "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0 :
         file_object.write("\n")
        file_object.write(it)
      
      with open ('averages.txt','r') as f :
        orders = [name.rstrip() for name in f]
        orders = [float(item) for item in orders]
        print(orders)

      torch.cuda.empty_cache()
      
      #delete output
      directory = os.getcwd()
      my_dir = directory
      for fname in os.listdir(my_dir):
       if fname.endswith(".png"):
        os.remove(os.path.join(my_dir, fname))
       if fname.endswith(".mp4"):
        os.remove(os.path.join(my_dir, fname))

@bot.command()
async def diffusion(ctx):
    if ctx.channel.id != channel_id:
      return
    gc.collect()
    torch.cuda.empty_cache()
    print('Command Loaded')
    print('Cache Prepared')
    async with ctx.channel.typing():
      #definitions
      author = ctx.message.author.id
      input = ctx.message.content
      prompt = input[11:len(input)]
      await bot.change_presence(activity=discord.Game(name=f"Diffusing {prompt}"))
      
      if os.environ['seed'] != '42':
       seed = os.environ['seed']
      if os.environ['seed'] == '0' or 0:
       seed = np.random.seed(0)
       seed = torch.seed()
       torch.manual_seed(seed)
      else:
       seed = np.random.seed(0)
       seed = torch.seed()
       torch.manual_seed(seed)
      print(seed)
      os.environ['seed'] = str(seed)
      print(seed)

      with open ('averages_diffusion.txt','r') as f :
        orders = [name.rstrip() for name in f]
        orders = [float(item) for item in orders]
        print(orders)
    
      average = sum(orders) / len(orders)
      average = str(round(average, 2))
      average = float(average)

      iterations = os.environ['diffusion_iterations']
      iterations = int(iterations)
      it_s = average
      epochs=1

      num = iterations/it_s
      num = num*epochs
      num_elapsed = iterations/it_s

      secs = 1
      if num > 420:
        num = 7
        secs = num_elapsed-420
      if num > 360:
        num = 6
        secs = num_elapsed-360
      if num > 300:
        num = 5 
        secs = num_elapsed-300
      if num > 240:
        num = 4
        secs = num_elapsed-240
      if num > 180:
        num = 3 
        secs = num_elapsed-180
      if num > 120:
        num = 2
        secs = num_elapsed-120
      if num > 60:
        num = 1
        secs = num_elapsed-60
      secs = int(secs)
      time.sleep(3)
      await ctx.channel.send("```Diffusing... ```"+f"**_{prompt}_**"+f'```Estimated: {num}m{secs}s Generation Time\nUsing seed: {seed}```')
      genTime = time.time()

      print(os.environ['prompt']) # outputs 'value'
      os.environ['prompt'] = prompt
      print(os.environ['prompt']) # outputs 'newvalue'

      subprocess.call(f'python3 diffusion.py')
      print('Generation Complete')
      
      new_prompt = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\progress_00000.png'
      upscale(new_prompt, 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\progress.png')
      await ctx.channel.send(f'<@{author}>'+'```Your Generation Is Done```')
      await ctx.channel.send(file=discord.File(new_prompt))
      ittime = time.time() - genTime
      print(f'Generation Time: {ittime}')
      elapsed = time.time() - genTime
      math_elapsed = time.time() - genTime
      secs = 1
      if elapsed > 420:
        elapsed = 7 
        secs = math_elapsed-420
      if elapsed > 360:
        elapsed = 6 
        secs = math_elapsed-360
      if elapsed > 300:
        elapsed = 5 
        secs = math_elapsed-300
      if elapsed > 240:
        elapsed = 4
        secs = math_elapsed-240
      if elapsed > 180:
        elapsed = 3 
        secs = math_elapsed-180
      if elapsed > 120:
        elapsed = 2
        secs = math_elapsed-120
      if elapsed > 60:
        elapsed = 1
        secs = math_elapsed-60
      print(f'Elapsed Time: {elapsed}')
      elapsed = str(elapsed)
      secs = int(secs)
      elapsed = f'{elapsed}m{secs}s'
      it = iterations / ittime
      it = "{:.2f}".format(it)
      it = str(it)
      in_loc = 'movie.mp4'
      out_loc = 'generated_out.mp4'
      clip = VideoFileClip(in_loc)
      print("fps: {}".format(clip.fps))
      clip = clip.set_fps(clip.fps * 2)
      final = clip.fx(vfx.speedx, 2)
      print("fps: {}".format(final.fps))
      final.write_videofile(out_loc)
      await ctx.channel.send(file=discord.File(out_loc))
      await ctx.channel.send(f'```Elapsed: {elapsed} | Generated at: {it}it/s```')

      with open("averages_diffusion.txt", "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0 :
         file_object.write("\n")
        file_object.write(it)
      
      with open ('averages_diffusion.txt','r') as f :
        orders = [name.rstrip() for name in f]
        orders = [float(item) for item in orders]
        print(orders)

      torch.cuda.empty_cache()
      
      #delete output
      directory = os.getcwd()
      my_dir = directory
      for fname in os.listdir(my_dir):
       if fname.endswith(".png"):
        os.remove(os.path.join(my_dir, fname))
       if fname.endswith(".mp4"):
        os.remove(os.path.join(my_dir, fname))
      await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

@bot.command()
async def faces(ctx):
 await bot.change_presence(activity=discord.Game(name=f"Finding Faces"))
 if ctx.message.attachments:
  if ctx.channel.id != channel_id:
    return
  async with ctx.channel.typing():
   link = ctx.message.attachments[0].url
   filename = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\urmom.png'
   r = requests.get(link, allow_redirects=True)
   open(filename, 'wb').write(r.content)
   print(filename)
   image = face_recognition.load_image_file(filename)
   face_locations = face_recognition.face_locations(image)

   for face_location in face_locations:
      top, right, bottom, left = face_location

      face_image = image[top:bottom, left:right]
      pil_image = Image.fromarray(face_image)
      pil_image.save('face.png')
      image = Image.open('face.png')
      new_image = image.resize((256, 256))
      new_image.save('face.png')
   test_image = face_recognition.load_image_file('face.png')
   face_locations = face_recognition.face_locations(test_image)
   face_encodings = face_recognition.face_encodings(test_image, face_locations)

   pil_image = Image.fromarray(test_image)
   draw = ImageDraw.Draw(pil_image)

   for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

     name = "Unknown Person"

     if True in matches:
       first_match_index = matches.index(True)
       name = known_face_names[first_match_index]

     draw.rectangle(((left, top), (right, bottom)), outline=(0,0,0))
     text_width, text_height = draw.textsize(name)
     draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
     draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255))

   del draw

   pil_image.save('identify.png')
   await ctx.reply(file=discord.File('identify.png'), mention_author=False)
   #delete output
   directory = os.getcwd()
   my_dir = directory
   for fname in os.listdir(my_dir):
    if fname.endswith(".png"):
      os.remove(os.path.join(my_dir, fname))
    if fname.endswith(".jpg"):
      os.remove(os.path.join(my_dir, fname))
   await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

@bot.command()
async def facehq(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Model Updating...')
      await ctx.channel.send('Updated! BATbot Model: `vqgan_faceshq`')
      os.environ['model'] = 'vqgan_faceshq'
      os.environ['height'] = '412'
      os.environ['width'] = '412'
      os.environ['max_iterations'] = '125'

@bot.command()
async def wikiart(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Model Updating...')
      await ctx.channel.send('Updated! BATbot Model: `wikiart_16384`')
      os.environ['model'] = 'wikiart_16384'
      os.environ['height'] = '288'
      os.environ['width'] = '512'
      os.environ['max_iterations'] = '400'

@bot.command()
async def default(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Model Updating...')
      await ctx.channel.send('Updated! BATbot Model: `vqgan_imagenet_f16_16384`')
      os.environ['model'] = 'vqgan_imagenet_f16_16384'
      os.environ['height'] = '412'
      os.environ['width'] = '412'
      os.environ['max_iterations'] = '400'

@bot.command()
async def d1024(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Model Updating...')
      await ctx.channel.send('Updated! BATbot Model: `vqgan_imagenet_f16_16384`')
      os.environ['model'] = 'vqgan_imagenet_f16_1024'
      os.environ['height'] = '412'
      os.environ['width'] = '412'
      os.environ['max_iterations'] = '400'

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
   open(filename, 'wb').write(r.content)
   print(filename)
   upscale(filename, 'progress_out.png')
   try:
    output_path = "progress_out.png"
    await ctx.reply(file=discord.File(output_path), mention_author=False)
   except FileNotFoundError as error:
    print(error)
    await ctx.channel.send("`Error: Image to large to be upscaled. Please try a smaller image.`")
   torch.cuda.empty_cache()
   await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

@bot.command()
async def rembg(ctx):
 await bot.change_presence(activity=discord.Game(name=f"Removing Background"))
 if ctx.message.attachments:
  if ctx.channel.id != channel_id:
    return
  async with ctx.channel.typing():
   link = ctx.message.attachments[0].url
   filename = link.split('/')[-1]
   r = requests.get(link, allow_redirects=True)
   open(filename, 'wb').write(r.content)
   print(filename)
   input_path = filename
   output_path = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\out.png'

   f = np.fromfile(input_path)
   result = remove(f)
   img = Image.open(io.BytesIO(result)).convert("RGBA")
   img.save(output_path)
   await ctx.reply(file=discord.File(output_path), mention_author=False)

   #delete output
   directory = os.getcwd()
   my_dir = directory
   for fname in os.listdir(my_dir):
    if fname.endswith(".png"):
      os.remove(os.path.join(my_dir, fname))
   torch.cuda.empty_cache()
   await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

@bot.command()
async def help(ctx):
   await bot.change_presence(activity=discord.Game(name=f"Helping"))
   if ctx.channel.id != channel_id:
     return
   print('Command Loaded')
   async with ctx.channel.typing():
     embed = discord.Embed(title="BATbot Help", description=f'`.rembg [Attached Image]`\n**removes background from attatched image**\n\n`.esrgan [Attatchment]`\n**BATbot will use a pretrained ESRGAN upscaler to upscale you images resolution by up to 4 times**\n\n`.status`\n**sends embed message with all relevent device stats for BATbot**\n\n`.imagine [Prompt]`\n**uses CLIP+VQGAN open generation to create an original image from your prompt**\n\n`.diffusion [Prompt]`\n**BATbot uses a CLIP+Diffusion model to generate images to match your prompt**\n\n`.facehq, .wikiart, .default, .d1024`\n**Changes BATbots VQGAN+CLIP model to one trained solely on faces, art or default configuration**\n\n`.square, .landscape, .portrait`\n**BATbot will update his size configurations for generations to your specified orientation**\n\n`.seed [Desired Seed]`\n**Changes BATbots seed for all open generation (if 0 will set to random)**\n\n`.gptj [Prompt]`\n**BATbot will use his trained GPT-J model to finish your prompt with natural language generation**\n\n`.sop [Attatchment]`\n**BATbot will turn your attatched image into a sequence of note lines ledgible by a computer, this allows BATbot to create a sound corolating to the "sounds of proccessing"**\n\n`.faces [Attatchment]`\n**BATbot will look through your photo and try to find any recognizable faces**\n\n__Any Attatchments Sent In This Channel Will Be Identified And Captioned By BATbot (To Prevent Captioning Include --nc In Your Message)__', color=0x7289da)
     await ctx.channel.send(embed=embed)
   torch.cuda.empty_cache()
   await bot.change_presence(activity=discord.Game(name=f"{randomstatus} sucks"))

@bot.command()
async def status(ctx):
    print('Command Loaded')
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
     name = (torch.cuda.get_device_name(0))
     available = torch.cuda.device_count()
     Used = '(Available)'
     if available == 1: 
        available =  f'There is currently `{available}` device available for generation'
        Used = '(Available)'

     if available == 2: 
        available =  f'There are currently `{available}` devices available for generation'

     if available == 0: 
        available =  f'There are currently `{available}` devices available for generation'
        Used = '(In Use)'
     status = (f'Using device: {device}\n{name} `{Used}`\n\n{available}\n\nBATbot is currently running on `{os.environ["model"]}`')

     embed = discord.Embed(title="Device Status", description=status, color=0x7289da) #,color=Hex code

     print(status)
     await ctx.channel.send(embed=embed)
     torch.cuda.empty_cache()

@bot.command()
async def kill(ctx):
    if ctx.channel.id != channel_id:
      return
    print('Command Loaded')
    async with ctx.channel.typing():
        await ctx.channel.send('```Exiting```')
        await bot.logout()
        sys.exit()

@bot.command()
async def seed(ctx):
   if ctx.channel.id != channel_id:
      return
   print('Command Loaded')
   async with ctx.channel.typing():
        input = ctx.message.content
        seed = input[6:len(input)]
        os.environ['seed'] = seed
        await ctx.channel.send(f'```Seed: {seed}```')
   torch.cuda.empty_cache()

@bot.command()
async def gptj(ctx):
   if ctx.channel.id != channel_id:
    return
   print('Command Loaded')
   async with ctx.channel.typing():
     input = ctx.message.content
     prompt = input[5:len(input)]
     max_length = 100
     temperature = 0.8
     top_probability = 0.9
     query = SimpleCompletion(prompt, length=max_length, t=temperature, top=top_probability)
     Query = query.simple_completion()
     embed = discord.Embed(title=f"**GPT-J-6B** `top_p={top_probability}, temp={temperature}`", description=f'**{prompt}**{Query}', color=0x7289da)
     await ctx.reply(embed=embed, mention_author=False)
   torch.cuda.empty_cache()

@bot.command()
async def sop(ctx):
 if ctx.message.attachments:
  if ctx.channel.id != channel_id:
    return
  async with ctx.channel.typing():
    #Download Image
    link = ctx.message.attachments[0].url
    filename = link.split('/')[-1]
    r = requests.get(link, allow_redirects=True)
    open(filename, 'wb').write(r.content)
    print(filename)

    input = filename
    os.environ['infile'] = input
    subprocess.call('python3 img_plt.py')
    output = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\out.wav'
    output2 = 'C:\\Users\\noahs\\Desktop\\BATbot\\Bot\\saved_figure.png'

    await ctx.reply(file=discord.File(output), mention_author=False)
    await ctx.channel.send(file=discord.File(output2), mention_author=False)
   
    #delete output
    directory = os.getcwd()
    my_dir = directory
    for fname in os.listdir(my_dir):
     if fname.endswith(".png"):
       os.remove(os.path.join(my_dir, fname))
     if fname.endswith(".wav"):
       os.remove(os.path.join(my_dir, fname))
    torch.cuda.empty_cache()

@bot.command(pass_context = True)
async def punish(ctx, member: discord.Member):
  channel = member.voice.channel
  voice = await channel.connect()
  bot.loop.create_task(play_source(voice))

@bot.command()
async def leave(ctx):
    await ctx.voice_client.disconnect()

@bot.command()
async def square(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Size Updating...')
      await ctx.channel.send('Updated! BATbot Size Config: `square`')
      os.environ['height'] = '412'
      os.environ['width'] = '412'

@bot.command()
async def portrait(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Size Updating...')
      await ctx.channel.send('Updated! BATbot Size Config: `portrait`')
      os.environ['height'] = '512'
      os.environ['width'] = '288'

@bot.command()
async def landscape(ctx):
    if ctx.channel.id != channel_id:
      return
    async with ctx.channel.typing():
      await ctx.channel.send('BAtbot Size Updating...')
      await ctx.channel.send('Updated! BATbot Size Config: `landscape`')
      os.environ['height'] = '288'
      os.environ['width'] = '512'

@bot.event
async def on_command_error(ctx, error):
    # if command has local error handler, return
    if hasattr(ctx.command, 'on_error'):
        return

    # get the original exception
    error = getattr(error, 'original', error)

    if isinstance(error, commands.CommandNotFound):
        return

    if isinstance(error, commands.BotMissingPermissions):
        missing = [perm.replace('_', ' ').replace('guild', 'server').title() for perm in error.missing_perms]
        if len(missing) > 2:
            fmt = '{}, and {}'.format("**, **".join(missing[:-1]), missing[-1])
        else:
            fmt = ' and '.join(missing)
        _message = 'I need the **{}** permission(s) to run this command.'.format(fmt)
        embed = discord.Embed(title=f"{ctx.command} error",
                              description='I need the **{}** permission(s) to run this command.'.format(fmt),
                              color=discord.Color.red())
        embed.set_footer(text=f"{error}")
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.DisabledCommand):
        embed = discord.Embed(title=f"{ctx.command} error",
                              description="This command has been disabled",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.CommandOnCooldown):
        remaining = "{}".format(str(datetime.timedelta(seconds=error.retry_after)))
        embed = discord.Embed(description=f"This command is on cooldown, please try again in "
                                          f"{remaining[0:1]} hours, "
                                          f"{remaining[3:4]} minutes, "
                                          f"{remaining[6:7]} seconds!\n"
                                          f"To avoid getting these cooldowns please vote by clicking above! This will "
                                          f"kick in within 1 minute and 30 seconds!",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if isinstance(error, discord.HTTPException):
        embed = discord.Embed(title=f"{ctx.command} error",
                              description=f"{error.text}",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.MissingPermissions):
        missing = [perm.replace('_', ' ').replace('guild', 'server').title() for perm in error.missing_perms]
        if len(missing) > 2:
            fmt = '{}, and {}'.format("**, **".join(missing[:-1]), missing[-1])
        else:
            fmt = ' and '.join(missing)
        _message = 'You need the **{}** permission(s) to use this command.'.format(fmt)
        embed = discord.Embed(title=f"{ctx.command} error",
                              description=f"{_message}",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    # This error is the most common and will need tweaking to how you setup your help command.
    if isinstance(error, commands.UserInputError):
        embed = discord.Embed(title=f"{ctx.command} error",
                              description=f"Invalid user input. "
                                          f"Please use `{bot.command_prefix}help {ctx.command.cog_name}` "
                                          f"and locate the `{ctx.command}` command. Check what arguments are "
                                          f"needed underneath it and retry this command!",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    if isinstance(error, commands.NoPrivateMessage):
        try:
            embed = discord.Embed(title=f"{ctx.command} error",
                                  description="This command cannot be sued in direct messages",
                                  color=discord.Color.red())
            await ctx.author.send(embed=embed)
        except discord.Forbidden:
            pass
        return

    if isinstance(error, commands.CheckFailure):
        embed = discord.Embed(title=f"{ctx.command} error",
                              description=f"You do not have permission to use this command",
                              color=discord.Color.red())
        await ctx.send(embed=embed)
        return

    # ignore all other exception types, but print them to stderr
    print('Ignoring exception in command {}:'.format(ctx.command), file=sys.stderr)

    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)

@bot.event
async def on_command(ctx):
    api.command_run(ctx)


from cogs.error_handling import ErrorHandling
from cogs.help import Help
from cogs.image_commands import ImageCommands
from cogs.natural_language_commands import NaturalLanguageCommands
from cogs.sound_commands import SoundCommands
from cogs.utils import UtilCommands

def load_cogs():
    bot.add_cog(ErrorHandling(bot))
    bot.add_cog(Help(bot))
    bot.add_cog(ImageCommands(bot))
    bot.add_cog(NaturalLanguageCommands(bot))
    bot.add_cog(SoundCommands(bot))
    bot.add_cog(UtilCommands(bot))
load_cogs()

bot.run(os.environ['bot_token'])