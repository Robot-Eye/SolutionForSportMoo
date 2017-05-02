#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: bodyPoseEstimation_V2.py
# Author: Eric Y. Huang <visualsolver@gmail.com>

from __future__ import print_function
import cv2
import csv
import re
import tqdm
import h5py
import tensorflow as tf
import numpy as np
import os, argparse
#import DataLayer as MPIIDataLayer
import MPIIDataLayer
import math
import random

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.gradproc import *

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 368, 368, 3), 'input'),
                InputVar(tf.float32, (None, 368, 368, 1), 'gmap'),
                InputVar(tf.float32, (None, 46, 46, 29), 'label'),
                ]

    def _build_graph(self, inputs):
        image, gmap, label = inputs
        gmap = tf.pad(gmap, [[0,0],[0,1],[0,1],[0,0]])
        pool_center = AvgPooling('mappool', gmap, 9, stride=8, padding='VALID')
        image = image*1.0/256 - 0.5
        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, 
            #W_init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)):
                W_init=tf.truncated_normal_initializer(stddev=0.01)):
            shared = (LinearWrap(image)
                .Conv2D('conv1_1', 64)
                .Conv2D('conv1_2', 64)
                .MaxPooling('pool1', 2)
                # 184
                .Conv2D('conv2_1', 128)
                .Conv2D('conv2_2', 128)
                .MaxPooling('pool2', 2)
                # 92
                .Conv2D('conv3_1', 256)
                .Conv2D('conv3_2', 256)
                .Conv2D('conv3_3', 256)
                .Conv2D('conv3_4', 256)
                .MaxPooling('pool3', 2)
                # 46
                .Conv2D('conv4_1', 512)
                .Conv2D('conv4_2', 512)
                .Conv2D('conv4_3_CPM', 256)
                .Conv2D('conv4_4_CPM', 256)
                .Conv2D('conv4_5_CPM', 256)
                .Conv2D('conv4_6_CPM', 256)
                .Conv2D('conv4_7_CPM', 128)())

        def add_stage(stage, l):
            l = tf.concat([l, shared, pool_center], 3, name='concat_stage{}'.format(stage))
            for i in range(1, 6):
                l = Conv2D('Mconv{}_stage{}'.format(i, stage), l, 128)
            l = Conv2D('Mconv6_stage{}'.format(stage), l, 128, kernel_shape=1)
            l = Conv2D('Mconv7_stage{}'.format(stage),
                    l, 29, kernel_shape=1, nl=tf.identity)
            return l

        with argscope(Conv2D, kernel_shape=7, nl=tf.nn.relu, 
            W_init=tf.truncated_normal_initializer(stddev=0.01)):
            out1 = (LinearWrap(shared)
                  .Conv2D('conv5_1_CPM', 512, kernel_shape=1)
                  .Conv2D('conv5_2_CPM', 29, kernel_shape=1, nl=tf.identity)())
            out2 = add_stage(2, out1)
            out3 = add_stage(3, out2)
            out4 = add_stage(4, out3)
            out5 = add_stage(5, out4)
            out6 = add_stage(6, out5)

        losses = []
        for idx, out in enumerate([out1, out2, out3, out4, out5, out6]):
            single_loss = tf.square(out - label)

            single_loss = tf.reduce_sum(single_loss, [1,2,3])
            single_loss = tf.reduce_mean(single_loss, name='loss{}'.format(idx+1) )
            # single_loss = symbolic_functions.print_stat(single_loss)
            losses.append(single_loss)
            #if idx == 1: break
        add_moving_summary(losses)

        # weight decay on all W of all layers
        wd_cost = tf.multiply(1e-5,
                         regularize_cost('.*/W', tf.nn.l2_loss),  #decay all weight
                         name='regularize_loss')
        add_moving_summary(wd_cost)
        losses.append(wd_cost)

        self.cost = tf.add_n(losses, name='overall_cost')

    def get_gradient_processor(self):
        return [ScaleGradient(
            [ ('.*stage2/.*', 3.0),
              ('.*stage3/.*', 3.0),
              ('.*stage4/.*', 4.0),
              ('.*stage5/.*', 4.0),
              ('.*stage6/.*', 4.0),]
            )]

def view_data(dataSource):
    ds = RepeatedData(get_data('train',dataSource), -1)
    ds.reset_state()
    for ims, gmaps, labels in ds.get_data():
        for im, gmap, label in zip(ims, gmaps, labels):
            print ('Gaussian Map max value: ' + str(gmap.max()))
            cv2.imshow("im", im*1.0/256)            
            for i in range(29):
                tmp = cv2.resize(label[:,:,i], (368,368) )
                print(np.max(label[:,:,i]))
                print(im.dtype)
                overlap = cv2.cvtColor(im.astype('float32'), cv2.COLOR_BGR2GRAY)*1.0/256*0.5 + tmp*0.5
                cv2.imshow("label", overlap)
                cv2.waitKey(1000)
                print ('Label Joint: ' + str(i) + ' has max value: ' + str(np.max(label)))

def get_data(name, dataSource):
    ds = None
    if name == 'train':
        ds = MPIIDataLayer.MPIIDataLayer([dataSource], '/home/user/tfpack-CPM/data/images' , 'train');
        augmentors = [
            imgaug.RandomApplyAug(imgaug.GaussianBlur(6), 0.5),     
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8,1.2), True),
            imgaug.RandomApplyAug(imgaug.JpegNoise(quality_range=(50, 100)), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianNoise(sigma=1), 0.8),
            ]
        ds = AugmentImageComponent(ds, augmentors)
        ds = PrefetchDataZMQ(ds,4) # use 20 threads pull data
        ds = BatchData(ds, 8)
    else:
        ds = MPIIDataLayer.MPIIDataLayer([dataSource],'/home/user/tfpack-CPM/data/images', 'val')
        ds = PrefetchDataZMQ(ds,10) # use 10 threads pull data
        ds = BatchData(ds, 2)
    return ds

def get_config(dataSource):
    logger.auto_set_dir()
    dataset_train = get_data('train',dataSource)
    step_per_epoch = 5000 # 30000/16 = 1875
    dataset_val = get_data('val',dataSource)

    lr = tf.Variable( 1e-2, trainable=False, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)
 
    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(0, 1e-6), (20, (1e-6)/3), (40, (1e-6)/9), (60, (1e-6)/27)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val, ScalarStats(['loss1', 'loss2', 'loss3','loss4','loss5','loss6']))
            ]),
        model=Model(),
        session_config=get_default_sess_config(0.9),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model, EX: ./vgg-19.npy')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--data_source', help='the source csv file')
    args = parser.parse_args()

    if args.run:
        #TODO: implement test
        #run_test(args.run)
        print ('not implemented yet, coming soon!')
        exit(0);
    elif args.view:
        view_data(args.data_source)
    else:
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        config = get_config(args.data_source)
        if args.load:
            # param = np.load('./vgg-19.npy').item()
            param = np.load(args.load).item()
            config.session_init = ParamRestore(param)
            print("**********************Welcome To Use Body Pose Estimation************************")
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        SyncMultiGPUTrainer(config).train()
        # SimpleTrainer(config).train()

