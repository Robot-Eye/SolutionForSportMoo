"""
Welcome to use  ROBOT EYE POSE ESTIMATION v0.9
ATTENTION: pre-frame based estimation & NO smoothing
USAGE: REQUEST: torch, torchx, torch-opencv
"""

import os

DATA_DIR = './data/'
RESULT_DIR = './result/'
VIDEO_NAME = 'IMG_1775_A'
VIDEO_PATH = DATA_DIR + VIDEO_NAME

if not os.path.exists(VIDEO_PATH):
    os.mkdir(VIDEO_PATH,0777)

if not os.path.exists(RESULT_DIR+VIDEO_NAME):
    os.mkdir(RESULT_DIR+VIDEO_NAME,0777)

# Cut input video into images
os.system('ffmpeg -i %s -r 30/1 %s' %(VIDEO_PATH+'.mp4',VIDEO_PATH+'/$'+VIDEO_NAME+'_%05d.png'))

# Remeber to modify the loation of person we want to estimate
# The output are images named pm_XXXXXX.png and a csv file which contains everything we want

os.system('th POSE_ESTIMATION.lua') 

os.system('ffmpeg -r 20 -i %s -pix_fmt yuv420p %s' %(RESULT_DIR+VIDEO_NAME+'/pm_%05d.png',RESULT_DIR+VIDEO_NAME+'.mp4'))


