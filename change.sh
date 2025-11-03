#!/bin/bash

# 삭제할 디렉토리 경로
BUILD_DIR="/home/jmseo1204/catkin_ws/build/git_fast_livo2"
SRC_DIR="/home/jmseo1204/catkin_ws/src/git_fast_livo2"

# 디렉토리 내용 삭제 함수
delete_directory_contents() {
  local TARGET_DIR=$1
  if [ -d "$TARGET_DIR" ]; then
    echo "Deleting all files in: $TARGET_DIR"
    rm -rf "$TARGET_DIR"/*
    rm -rf "$TARGET_DIR"/.[!.]* "$TARGET_DIR"/..?*
    echo "Done."
  else
    echo "Directory not found: $TARGET_DIR"
  fi
}

# build 디렉토리 삭제
delete_directory_contents "$BUILD_DIR"

# src 디렉토리 삭제
delete_directory_contents "$SRC_DIR"