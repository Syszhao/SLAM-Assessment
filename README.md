# SLAM-Assessment
extern里面是使用的五种slam方法，针对这几种方法有如下用法：
## 单目SLAM评估（Egodex数据集）
1. egodex_megasam_rerun.py是Megasam的rerun可视化，把文件放在Megasam目录下
2. egodex_rerun.py是Droid-slam的rerun可视化，把文件放在Droid-slam目录下
3. egodex_vo.py是Macvo的rerun可视化，把文件放在Macvo目录下
4. egodex这个文件是Cuvslam的用法，把文件放在Cuvslam/example目录下
5. egodex_all_slam_rerun.py是总启动文件，放在主目录下
## 双目SLAM评估（Lerobot数据集）
1. lerobotv3这个文件是Cuvslam的用法，把文件放在Cuvslam/example目录下
2. lerobot_stereo_macvo.py是Macvo的rerun可视化，把文件放在Macvo目录下
3. lerobot_stereo_orbslam.py是Orbslam的rerun可视化，把文件放在Orbslam目录下
4. lerobot_stereo_megasam.py是Megasam的rerun可视化，把文件放在Megasam目录下
5. lerobot_v3_common.py、lerobot_all_slam_rerun.py是总启动文件，放在主目录下
## Megasam手部Mask可视化
1. test_demo.py替换原来的Megasam/camera_tracking_scripts下的test_demo.py文件
2. xperience_megasam_pipeline.py是启动文件，放在主目录下
