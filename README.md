# SolutionForSportMoo



"**********************Welcome To Use Body Pose Estimation************************"


Requirements: software

	-- Python 2 or 3
	-- TensorFlow >= 1.0.0
	-- Python bindings for OpenCV	
	-- matlab
	-- Tensorpack 
		A Neural Network Toolbox on TensorFlow. 
		We build our model on this platform, you can get it from https://github.com/ppwwyyxx/tensorpack
	
Requirements: hardware

	-- GPU: Titan, Titan Black, Titan X, K20, K40, K80, GTX 1080Ti.
   		At least 8GB GPU memory will be needed.
   
Preparation for Testing:

1) Cut input video into images.

	Using ffmpeg

	Ex.
	$ffmpeg -i IMG_1773_A.mp4 -r 30/1 $IMG_1779_A_%05d.png


2) Determine the bounding box for person. This model relies a good person detection and a "good" bounding box, the input is only part of image that contains the person we want to estimate.


Thus, first of all, change line 31 in script: run_cpm.lua

      RGB_image = image.crop(RGB_image,19,400,19+480,400+480)
			left corner of bounding box(x,y), right corner of bounding box (x,y)

	You have to manually estimate the position of bounding box and place it here.
	(assume person is stationary through the video

3) Change the input dir:

  (the folder contains all the images that output from ffmpeg. It has to be png


4) run cpm:

Since the CPM is implemented in torch, please run CPM using commandline below

	$th run_cpm.lua

The output will be run_cpm%5d.png in current folder and also a .csv file will record all pose information


5) Using ffmpeg to put all output images back to video:

	ffmpeg -r 20 -i new_%05d.png -pix_fmt yuv420p test.mp4


Preparation for Training:

Data:
Download MPII Human Pose Dataset(http://human-pose.mpi-inf.mpg.de/#overview) to ./train/data/
As the annotations are stored in a matlab structure, we need to run mat2CSV.m to transform mpii_human_pose_v1_u12_1.mat to data.csv we need.


Training:

Run train/RobotEye_BodyPoseEstimation_V2.py :

	Ex.
	$cd train
	$python ./RobotEye_BodyPoseEstimation_V2.py --gpu 0,1 --data_source ./data/data.csv

(Remember to change the data path)

Note: The training time of one epoch is ~40 min on GTX 1080Ti.
      The entire training time will cost one and a half week.
      You can use tensorboard to watch the details of training.
      The log of training can be find in ./train_log

Any questions you can contact us:

Eric: 	visualsolver@gmail.com
yangzl: yangzl15@mails.tsinghua.edu.cn


