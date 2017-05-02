from tensorpack import *
import cv2
import numpy as np
import sys
import glob
import errno
import os
import csv
import random
import h5py


from numpy import linalg as LA
import math
import re

			

def drawGaussianBlob(gaussian_map, sigma, pt_x, pt_y ):
	for x_p in range(gaussian_map.shape[1]):
		for y_p in range(gaussian_map.shape[0]):
			dist_sq = (x_p - pt_x) * (x_p - pt_x) + (y_p - pt_y) * (y_p - pt_y)
			exponent = dist_sq / 2.0 / sigma / sigma
			gaussian_map[y_p, x_p] = math.exp(-exponent)
	return gaussian_map

def getJetColor(v, vmin, vmax):
	c = np.zeros((3))
	if (v < vmin):
		v = vmin
	if (v > vmax):
		v = vmax
	dv = vmax - vmin
	if (v < (vmin + 0.125 * dv)): 
		c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
	elif (v < (vmin + 0.375 * dv)):
		c[0] = 255
		c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
	elif (v < (vmin + 0.625 * dv)):
		c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
		c[1] = 255
		c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
	elif (v < (vmin + 0.875 * dv)):
		c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
		c[2] = 255
	else:
		c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5					  
	return c

def viz(img, prediction, occulsions):
	limbs = np.array([[0,1,2,   3,4,5,   6,7,8,	   9,10,11,  12,13,14,  15,16,17,18  ],
					  [1,2,18,  4,5,17,  7,8,16,  10,11,15,  13,14,19,  20,21,22,23  ]])
	stickwidth = 2
	colors = [[0, 0, 255],[0, 0, 255],[0, 0, 255],
			  [0, 170, 255],[0, 170, 255],[0, 170, 255],
			  [0, 255, 170],[0, 255, 170],[0, 255, 170],
			  [0, 255, 0],[0, 255, 0],[0, 255, 0],
			  [170, 255, 0],[170, 255, 0],[170, 255, 0],
			  [255, 170, 0],[255, 170, 0],[255, 170, 0],[255, 170, 0]] # note BGR ...

	canvas = img.copy()
	cur_canvas = canvas.copy()
	
	for part in range(24):
		if occulsions[part] < 0.2:
			cv2.circle(cur_canvas, (int(prediction[part, 1]), int(prediction[part, 0])), int(max(occulsions[part]*10, 5)), (0,0,0), -1)
			# cv2.circle(cur_canvas, (int(prediction[part, 1]), int(prediction[part, 0])), 5, getJetColor(occulsions[part],0,1), -1)
		else:
			cv2.circle(cur_canvas, (int(prediction[part, 1]), int(prediction[part, 0])), int(max(occulsions[part]*10, 5)), (0,0,255), -1)
			# cv2.circle(cur_canvas, (int(prediction[part, 1]), int(prediction[part, 0])), 5, getJetColor(occulsions[part],0,1), -1)

	for l in range(limbs.shape[1]):


		X = prediction[limbs[:,l], 0] #row
		Y = prediction[limbs[:,l], 1] #col
		V = [occulsions[limbs[:,l][0]], occulsions[limbs[:,l][1]]]
		# print(X)
		# print(Y)
		# print(V)
		mX = np.mean(X)
		mY = np.mean(Y)
		length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
		angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
		polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
		if V[0] < 0.2 or V[1] < 0.2:
			cv2.fillConvexPoly(cur_canvas, polygon, (0, 0, 0))
		else:
			cv2.fillConvexPoly(cur_canvas, polygon, colors[l])
	canvas = canvas * 0.5 + cur_canvas * 0.5 # for transparency
	return canvas



JOINT_NAMES = [
				'head',
				'neck',
				'left_shoulder','right_shoulder',
				'left_elbow','right_elbow',
				'left_hand','right_hand',
				'left_hip','right_hip',
				'left_knee','right_knee',
				'left_foot','right_foot'
				]


class MPIIDataLayer(DataFlow):
	def __init__(self, csv_dir_list, data_folder, name):
		self.gt_csv = csv_dir_list
		self.data_folder = data_folder
		# get gt
		self.gt = list()
		self.trainOrVal = name
		train_list = list()
		val_list = list()
		for csv_dir in csv_dir_list:
			print(csv_dir)
			counter = 0
			with open(csv_dir) as csvfile:
				reader = csv.DictReader(csvfile)
				data_entry = None
				for row in reader:
					data_entry = dict()
					# 1. add file_name
					file_name = row['image_absolutePath']
					data_entry['filename'] = file_name;
					# 2. add trainOrVal
					data_entry['trainOrVal'] = int(row['trainOrVal'])
					# 3. add bounding box
					if row['bounding_box_left_up_corner_x'] == '':
						continue
					lu_x = float(row['bounding_box_left_up_corner_x'])
					lu_y = float(row['bounding_box_left_up_corner_y'])
					rd_x = float(row['bounding_box_right_down_corner_x'])
					rd_y = float(row['bounding_box_right_down_corner_y'])
					data_entry['boundingbox'] = np.asarray([float(lu_x), float(lu_y), float(rd_x), float(rd_y)])
					# 4. add 14 joint ground truth
					body_joint = np.zeros((14,3))
					skip = False
					for i, f_ in enumerate(JOINT_NAMES):
						if row[f_+'_x'] == '' or row[f_+'_y'] == '':
							skip = True
							continue # ingnore incomplete data
						body_joint[i][0] = float(row[f_+'_x'])
						body_joint[i][1] = float(row[f_+'_y'])
						body_joint[i][2] = int(row[f_+'_visiblity'])
					data_entry['label'] = body_joint
					counter = counter + 1
					if skip == False and row['trainOrVal'] == '0':
						train_list.append(data_entry)	
					elif skip == False:
						val_list.append(data_entry)
		if self.trainOrVal == 'train':
			self.gt = train_list
		else:
			self.gt = val_list
		print('MPII total < ' + self.trainOrVal + ' > data size: ' + str(counter))


	def get_data(self):
		random.shuffle(self.gt)
		for counter, kkk in enumerate(self.gt):
			image_file_name = os.path.join(self.data_folder, kkk['filename'])
			img = cv2.imread(image_file_name,  cv2.IMREAD_UNCHANGED)
			bbox_width = int(kkk['boundingbox'][2]) - int(kkk['boundingbox'][0])
			bbox_height = int(kkk['boundingbox'][3]) - int( kkk['boundingbox'][1])
			img_data = np.ones((bbox_height, bbox_width, 3)) * 128

			left_corner_y = max(0, int(kkk['boundingbox'][1]))
			left_corner_x = max(0, int(kkk['boundingbox'][0]))
			right_corner_y = min(img.shape[0], int(kkk['boundingbox'][3]))
			right_corner_x = min(img.shape[1], int(kkk['boundingbox'][2]))
			roi = img[left_corner_y:right_corner_y, left_corner_x:right_corner_x]
			assert abs(bbox_height - bbox_height) < 1.0
			shift_x = 0
			shift_y = 0
			if int(kkk['boundingbox'][0]) < 0:
				shift_x = - int(kkk['boundingbox'][0])
			if int(kkk['boundingbox'][1]) < 0:
				shift_y =  - int(kkk['boundingbox'][1])
			img_data[shift_y:shift_y + roi.shape[0], shift_x:shift_x + roi.shape[1]] = roi
			roi = cv2.resize(roi, ( 368, 368))
			
			gt_tensor = np.zeros((46,46, 14*2 + 1)).astype('float32')
			for i in range(0,14):
				pt_x = int(kkk['label'][i][0]/8)
				pt_y = int(kkk['label'][i][1]/8)
				#Visible
				if int(kkk['label'][i][2]) == 1 :
					gt_tensor[:,:,i] = drawGaussianBlob(gt_tensor[:,:,i], 1.5, pt_x, pt_y )
				#Occulusion
				else:
					gt_tensor[:,:,i+14] = drawGaussianBlob(gt_tensor[:,:,i+14], 1.5, pt_x, pt_y )

			gt_tensor[:,:,14*2] = 1 - np.amax(gt_tensor[:,:,0:24*2], 2)
			gmap = np.zeros((368,368,1)).astype('float32')
			gmap = drawGaussianBlob(gmap, 25, 368/2, 368/2 )
			yield [roi, gmap, gt_tensor]

	def size(self):
		return len(self.gt)

if __name__ == '__main__':
	a = MPIIDataLayer(['/home/user/tfpack-CPM/data/data.csv'], '/home/user/tfpack-CPM/data/images', 'train')
	print(a.size())
	a = MPIIDataLayer(['/home/user/tfpack-CPM/data/data.csv'], '/home/user/tfpack-CPM/data/images', 'val')
	print(a.size())
	
	
