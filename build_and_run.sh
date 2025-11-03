#!/bin/bash

cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
# roslaunch fast_livo mapping_snu.launch
# roslaunch fast_livo mapping_hesaixt32_hilti22.launch
roslaunch fast_livo mapping_newer.launch