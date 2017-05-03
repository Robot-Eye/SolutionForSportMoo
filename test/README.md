# SolutionForSportMoo



"**********************Welcome To Use Body Pose Estimation************************"


Preparation for Testing:

1) Cut input video into images.

Using ffmpeg

Ex. 
$cd data
$mkdir IMG_1775_A
$ffmpeg -i IMG_1775_A.mp4 -r 30/1 ./IMG_1775_A/$IMG_1775_A_%05d.png


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

Any questions you can contact us:

Eric: 	visualsolver@gmail.com
yangzl: yangzl15@mails.tsinghua.edu.cn


