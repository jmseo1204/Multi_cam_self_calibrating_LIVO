# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import io
import os

# 1. 입력 파일에서 데이터 로드
file_path = os.path.join(os.path.dirname(__file__), "results", "1125_7534_stat.txt")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

df = pd.read_csv(file_path, comment="#", header=None, sep=",\s*", engine="python")

# 시간 축 변환: x := log(t - t0 + 1)
# t0은 데이터셋에서 가장 먼저 찍힌 시간
try:
    t0 = float(df.iloc[:, 0].min())
except Exception:
    t0 = 0.0
# 열 이름 지정 전에 임시 0열을 time으로 가정하여 변환 열 생성
df["time_shift"] = np.log(np.clip(df.iloc[:, 0] - t0 + 1.0, a_min=1e-9, a_max=None))

# 2. 데이터프레임 컬럼명 설정
# 헤더를 기반으로 컬럼명을 생성합니다.
columns = ["time"]
columns += [f"cam_used_{i}" for i in range(4)]
columns += [f"cam_rows_{i}" for i in range(4)]
for i in range(4):
    columns += [f"cam_sol_{i}_{j}" for j in range(7)]
for i in range(4):
    columns.append(f"cam_rot_std_{i}")
    columns.append(f"cam_trans_std_{i}")
columns += ["total_rot_std", "total_trans_std"]
columns.append("level0_avg_total_rows")
columns += ["rot_max_eig_post", "trans_max_eig_post"]
for i in range(4):
    columns.append(f"cam_patch_count_{i}")

# 파일 내 컬럼 수와 맞추기 위해 초과 컬럼명은 자르고, 부족하면 남기는 대로 둡니다.
if len(columns) >= df.shape[1]:
    df.columns = columns[: df.shape[1]]
else:
    # 파일이 더 많은 열을 가진 경우, 나머지 열에 대해 generic 이름 부여
    extra = [f"extra_{k}" for k in range(df.shape[1] - len(columns))]
    df.columns = columns + extra

# time_shift 열을 컬럼명 정리 이후에도 유지
if "time_shift" not in df.columns:
    # 재계산 (정리된 time 컬럼 사용)
    t0 = float(df["time"].min())
    df["time_shift"] = np.log(np.clip(df["time"] - t0 + 1.0, a_min=1e-9, a_max=None))


# 3. 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # If either vector is zero, return 0 as similarity (or define differently)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


# 4. 각 행을 순회하며 유사도 계산 (rotation과 translation 분리)
rotation_results = []
translation_results = []
for index, row in df.iterrows():
    # Extract cam_sol vectors and keep non-zero vectors
    cam_sols = {}
    for i in range(4):
        cols_needed = [c for c in [f"cam_sol_{i}_{j}" for j in range(7)] if c in df.columns]
        if len(cols_needed) < 7:
            continue
        vec = row[cols_needed].to_numpy(dtype=float)
        # Keep only if not a zero vector (consider floating precision)
        if np.linalg.norm(vec) > 1e-9:
            cam_sols[i] = vec

    # Proceed only when there are at least two non-zero vectors
    if len(cam_sols) >= 2:
        # Compute cosine similarity for all vector pairs
        for i, j in combinations(cam_sols.keys(), 2):
            v1 = cam_sols[i]
            v2 = cam_sols[j]

            # Separate rotation (first 3 components) and translation (next 3 components)
            rot1, rot2 = v1[:3], v2[:3]
            trans1, trans2 = v1[3:6], v2[3:6]

            # Compute similarities for rotation and translation
            rot_similarity = cosine_similarity(rot1, rot2)
            trans_similarity = cosine_similarity(trans1, trans2)
            if rot_similarity != 0:
                rotation_results.append({"time": row["time"], "pair": f"cam{i}-cam{j}", "similarity": rot_similarity})
            if trans_similarity != 0:
                translation_results.append(
                    {"time": row["time"], "pair": f"cam{i}-cam{j}", "similarity": trans_similarity}
                )

# 5. 결과 데이터프레임 생성
rotation_df = pd.DataFrame(rotation_results)
translation_df = pd.DataFrame(translation_results)


# 6. 4x4 그리드 생성 (평균 유사도)
def create_similarity_grid(results_df, title):
    """Create 4x4 grid showing average cosine similarity between camera pairs."""
    # Initialize 4x4 matrix
    grid = np.zeros((4, 4))

    # Fill diagonal with 1.0 (self-similarity)
    np.fill_diagonal(grid, 1.0)

    # Calculate average similarities for each pair
    for i in range(4):
        for j in range(i + 1, 4):
            pair_name = f"cam{i}-cam{j}"
            pair_data = results_df[results_df["pair"] == pair_name]
            if not pair_data.empty:
                avg_sim = pair_data["similarity"].mean()
                grid[i, j] = avg_sim
                grid[j, i] = avg_sim  # Symmetric

    return grid


# Create grids for rotation and translation
rotation_grid = create_similarity_grid(rotation_df, "Rotation")
translation_grid = create_similarity_grid(translation_df, "Translation")

# 7. 시각화 (Figure 1: Similarity & Distributions)
plt.style.use("seaborn-v0_8-whitegrid")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Figure 1: Cosine Similarity Analysis", fontsize=16)

# 7-1. Rotation similarity grid
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(
    rotation_grid,
    annot=True,
    cmap="RdBu_r",
    center=0,
    xticklabels=[f"cam{i}" for i in range(4)],
    yticklabels=[f"cam{i}" for i in range(4)],
    ax=ax1,
    vmin=-1,
    vmax=1,
)
ax1.set_title("Average Rotation Similarity (4x4 Grid)")

# 7-2. Translation similarity grid
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(
    translation_grid,
    annot=True,
    cmap="RdBu_r",
    center=0,
    xticklabels=[f"cam{i}" for i in range(4)],
    yticklabels=[f"cam{i}" for i in range(4)],
    ax=ax2,
    vmin=-1,
    vmax=1,
)
ax2.set_title("Average Translation Similarity (4x4 Grid)")

# 7-3. Rotation similarity distribution
ax3 = plt.subplot(2, 3, 3)
if not rotation_df.empty:
    sns.histplot(data=rotation_df, x="similarity", kde=True, ax=ax3, bins=20)
    ax3.set_title("Rotation Similarity Distribution")
    ax3.set_xlabel("Cosine similarity")
    ax3.set_ylabel("Count")
    ax3.set_xlim(-1.1, 1.1)

# 7-4. Translation similarity distribution
ax4 = plt.subplot(2, 3, 4)
if not translation_df.empty:
    sns.histplot(data=translation_df, x="similarity", kde=True, ax=ax4, bins=20)
    ax4.set_title("Translation Similarity Distribution")
    ax4.set_xlabel("Cosine similarity")
    ax4.set_ylabel("Count")
    ax4.set_xlim(-1.1, 1.1)

# 7-5. Rotation similarity over time
ax5 = plt.subplot(2, 3, 5)
if not rotation_df.empty:
    # rotation_df의 time을 변환 시간축으로 매핑
    rot_plot = rotation_df.copy()
    rot_plot = rot_plot.merge(df[["time", "time_shift"]], on="time", how="left")
    sns.scatterplot(data=rot_plot, x="time_shift", y="similarity", hue="pair", style="pair", s=100, ax=ax5)
    ax5.set_title("Rotation Similarity over log(t - t0 + 1)")
    ax5.set_xlabel("log(t - t0 + 1)")
    ax5.set_ylabel("Cosine similarity")
    ax5.set_ylim(-1.1, 1.1)
    ax5.legend(title="Vector pair", bbox_to_anchor=(1.02, 1), loc="upper left")

# 7-6. Translation similarity over time
ax6 = plt.subplot(2, 3, 6)
if not translation_df.empty:
    trans_plot = translation_df.copy()
    trans_plot = trans_plot.merge(df[["time", "time_shift"]], on="time", how="left")
    sns.scatterplot(data=trans_plot, x="time_shift", y="similarity", hue="pair", style="pair", s=100, ax=ax6)
    ax6.set_title("Translation Similarity over log(t - t0 + 1)")
    ax6.set_xlabel("log(t - t0 + 1)")
    ax6.set_ylabel("Cosine similarity")
    ax6.set_ylim(-1.1, 1.1)
    ax6.legend(title="Vector pair", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

# 8. Figure 2: Update Standard Deviations & Eigenvalues & Patch Counts
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle("Figure 2: Update Standard Deviations, Covariance Eigenvalues & Patch Counts", fontsize=16)

# 8-1. Rotation Update Std Dev & Max Eigenvalue
ax_rot = plt.subplot(3, 1, 1)
# Plot per-camera std devs
for i in range(4):
    col_name = f"cam_rot_std_{i}"
    if col_name in df.columns:
        ax_rot.plot(df["time_shift"], df[col_name], label=f"Cam {i} Std", alpha=0.6, linewidth=1)

# Plot total std dev
if "total_rot_std" in df.columns:
    ax_rot.plot(df["time_shift"], df["total_rot_std"], label="Total Std", color="black", linewidth=2, linestyle="--")

ax_rot.set_xlabel("log(t - t0 + 1)")
ax_rot.set_ylabel("Update Std Dev (Rotation)")
ax_rot.set_title("Rotation Update Std Dev & Max Eigenvalue")
ax_rot.legend(loc="upper left")

# Overlay Max Eigenvalue on twin axis
if "rot_max_eig_post" in df.columns:
    ax_rot2 = ax_rot.twinx()
    ax_rot2.plot(df["time_shift"], df["rot_max_eig_post"], color="red", label="Max Eig Post", linewidth=2, alpha=0.8)
    ax_rot2.set_ylabel("Max Eigenvalue (Rotation)")
    ax_rot2.legend(loc="upper right")

# 8-2. Translation Update Std Dev & Max Eigenvalue
ax_trans = plt.subplot(3, 1, 2)
# Plot per-camera std devs
for i in range(4):
    col_name = f"cam_trans_std_{i}"
    if col_name in df.columns:
        ax_trans.plot(df["time_shift"], df[col_name], label=f"Cam {i} Std", alpha=0.6, linewidth=1)

# Plot total std dev
if "total_trans_std" in df.columns:
    ax_trans.plot(df["time_shift"], df["total_trans_std"], label="Total Std", color="black", linewidth=2, linestyle="--")

ax_trans.set_xlabel("log(t - t0 + 1)")
ax_trans.set_ylabel("Update Std Dev (Translation)")
ax_trans.set_title("Translation Update Std Dev & Max Eigenvalue")
ax_trans.legend(loc="upper left")

# Overlay Max Eigenvalue on twin axis
if "trans_max_eig_post" in df.columns:
    ax_trans2 = ax_trans.twinx()
    ax_trans2.plot(df["time_shift"], df["trans_max_eig_post"], color="red", label="Max Eig Post", linewidth=2, alpha=0.8)
    ax_trans2.set_ylabel("Max Eigenvalue (Translation)")
    ax_trans2.legend(loc="upper right")

# 8-3. Per-Camera Patch Counts
ax_patch = plt.subplot(3, 1, 3)
for i in range(4):
    col_name = f"cam_patch_count_{i}"
    if col_name in df.columns:
        ax_patch.plot(df["time_shift"], df[col_name], label=f"Cam {i} Patch Count", alpha=0.7, linewidth=1.5)

ax_patch.set_xlabel("log(t - t0 + 1)")
ax_patch.set_ylabel("Patch Count")
ax_patch.set_title("Patch Count over Time")
ax_patch.legend(loc="upper right")
ax_patch.grid(True)

plt.tight_layout()
plt.show()

# 9. Print summary statistics
if not rotation_df.empty and not translation_df.empty:
    print("--- Analysis Results ---")
    print("\nRotation Similarity Statistics:")
    print(rotation_df.groupby("pair")["similarity"].agg(["mean", "std", "count"]))
    print("\nTranslation Similarity Statistics:")
    print(translation_df.groupby("pair")["similarity"].agg(["mean", "std", "count"]))

    print("\nAverage Similarity Matrices:")
    print("Rotation:")
    print(rotation_grid)
    print("\nTranslation:")
    print(translation_grid)
else:
    print("No data to analyze. (No rows with at least two non-zero cam_sol vectors)")
