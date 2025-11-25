#!/bin/bash

OLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 복사 대상 디렉토리
DEST=~/catkin_ws/src/git_fast_livo2

# 복사 대상 디렉토리 초기화
mkdir -p "$DEST"

# Git에서 추적 중인 파일만 복사
cd "$OLD_DIR"
git ls-files -z | rsync -av --files-from=- --from0 ./ "$DEST/"

# 빌드
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash

cd "$OLD_DIR"
