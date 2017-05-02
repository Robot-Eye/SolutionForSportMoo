

"**********************Welcome To Use Body Pose Estimation************************"

Preparation for Training:

Data:
Download MPII Human Pose Dataset(http://human-pose.mpi-inf.mpg.de/#overview) to ./train/data/ .

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


