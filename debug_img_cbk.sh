#!/bin/bash

echo "=== img_cbk 호출 안 되는 문제 디버깅 ==="
echo ""

echo "1. republish 노드가 실행 중인지 확인:"
rosnode list | grep republish
echo ""

echo "2. 이미지 토픽이 실제로 publish되는지 확인:"
echo "   - 압축된 토픽 (rosbag에서 publish):"
rostopic list | grep alphasense_driver_ros.*compressed
echo "   - 압축 해제된 토픽 (republish 노드가 publish):"
rostopic list | grep "^/alphasense_driver_ros/cam[0-9]"
echo ""

echo "3. 토픽 타입 확인:"
for topic in /alphasense_driver_ros/cam0 /alphasense_driver_ros/cam1 /alphasense_driver_ros/cam3 /alphasense_driver_ros/cam4; do
  echo "   $topic:"
  rostopic type $topic 2>/dev/null || echo "     (토픽이 존재하지 않음)"
done
echo ""

echo "4. 토픽 publish 상태 확인 (Hz):"
for topic in /alphasense_driver_ros/cam0 /alphasense_driver_ros/cam1 /alphasense_driver_ros/cam3 /alphasense_driver_ros/cam4; do
  echo "   $topic:"
  timeout 2 rostopic hz $topic 2>/dev/null || echo "     (토픽이 publish되지 않음)"
done
echo ""

echo "5. laserMapping 노드가 구독 중인 토픽 확인:"
rosnode info /laserMapping 2>/dev/null | grep -A 20 "Subscriptions:" || echo "   (노드를 찾을 수 없음)"
echo ""

echo "6. img_en 설정 확인:"
rosparam get /common/img_en 2>/dev/null || echo "   (파라미터를 찾을 수 없음)"
echo ""

echo "7. 토픽 연결 상태 확인:"
rostopic info /alphasense_driver_ros/cam0 2>/dev/null | grep -A 5 "Subscribers:" || echo "   (토픽이 존재하지 않음)"


