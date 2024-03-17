# mywolk --------v1--------
import gradio as gr

import torch
from PIL import Image
import clip
import os.path as osp
import os, sys
sys.path.insert(0, '../')
sys.path.append('.')
sys.path.append('..')
import shutil ###

import torchvision.utils as vutils
from lib.utils import load_model_weights,mkdir_p
from models.GALIP_new import NetG, CLIP_TXT_ENCODER
import argparse
import cv2
import glob
import numpy as np
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import gradio as gr 


device = 'cpu' # 'cpu' # 'cuda:0'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()


#清空
# 清空文件夹
def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        # 清空
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


mkdir_p('./samples')
folder_path = './samples' # 清空
clear_folder(folder_path)
# 输入语料
# captions = ['She is young, attractive,\
#              and slim with quite rounded nose,\
#              no bangs, straight eyebrows, \
#             quite small lips. She has black hair. \
#             She has a normal chin, no beard. \
#              She does not put neither lipstick, \
#             neck tie, necklace, earrings, heavy makeup, hat nor eyeglasses.']

def generate_images(captions , batch_size):
    text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)
    netG = NetG(64, 150, 512, 256, 3, False, clip_model).to(device)
    path = './saved_models/state_epoch_360.pth'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)
    # batch_size = 8 #需要生成多少张图片
    noise = torch.randn((batch_size, 150)).to(device)
    # captions = prompt
    
    # 推理
    with torch.no_grad():
        for i in range(len(captions)):
            caption = captions[i]
            tokenized_text = clip.tokenize([caption]).to(device)
            sent_emb, word_emb = text_encoder(tokenized_text)
            sent_emb = sent_emb.repeat(batch_size, 1)
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            # batch_size = 1 #需要生成多少张图片
            # 分开保存
            for j in range(batch_size):
                fake_img = fake_imgs[j]
                name = 'demo_%d' % (i * batch_size + j + 1)
                vutils.save_image(fake_img, './samples/%s.png' % (name), value_range=(-1, 1), normalize=True)



# chao分
def img_sr(img_faces):
    # img = 
    # base64_decoded = base64.b64decode(img_base64)
    # byte_stream = io.BytesIO(base64_decoded)
    # pil_image = Image.open(byte_stream)
    return 


captions = gr.Textbox(lines=5, label="输入Captions")
batch_size = gr.Number(label="Batch Size")
image_output = gr.Image(label="生成的图片")

def generate_images_interface(captions, batch_size):
    generate_images(captions, batch_size)
    return f'./samples/demo_1.png'  # 返回生成的图片路径（这里假设只生成一张图片）

iface = gr.Interface(
    fn=generate_images_interface,
    inputs=[captions, batch_size],
    outputs=image_output,
    title="Generate Images",
    description="输入Captions和Batch Size以生成图片"
)

iface.launch()
