EST_FILE="$1"
GT_FILE="results/exp18_gt.txt"

# 확장자 제거 (예: results/2164.txt → results/2164)
# EST_BASE="${EST_FILE%.*}"
# RESULT_FILE="${EST_BASE}.txt"

evo_ape tum "$GT_FILE" "$EST_FILE" --t_max_diff 0.03 -asvp