/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "feature.h"
#include "utils/types.h"
#include "voxel_map.h"
#include <cmath>
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <unsupported/Eigen/MatrixFunctions>
#include <vikit/math_utils.h>
#include <vikit/pinhole_camera.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>

struct SubSparseMap {
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<vector<float>> pixel_var; // 픽셀별 분산 벡터 추가
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap() {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    pixel_var.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset() {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    pixel_var.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp {
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix)
      : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS {
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() {
    for (VisualPoint *vp : voxel_points) {
      if (vp != nullptr) {
        delete vp;
        vp = nullptr;
      }
    }
  }
};

class VIOManager {
public:
  std::vector<Frame *> new_frame_vec;
  std::vector<SubSparseMap *> visual_submaps;
  std::vector<vk::AbstractCamera *> m_cameras;
  std::vector<M3D> m_R_c_i_vec;
  std::vector<V3D> m_P_c_i_vec;
  std::vector<M3D> m_R_c_l_vec;
  std::vector<V3D> m_P_c_l_vec;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  // int cam_idx;

  vector<unordered_map<VOXEL_LOCATION, int> *> sub_feat_maps;

  std::vector<double> m_img_time_offsets_from_last_update;

  std::vector<M3D> original_Rci_vec, original_Rcl_vec;
  std::vector<V3D> original_Pci_vec, original_Pcl_vec;
  // M3D original_Jdphi_dR, original_Jdp_dR;

  int min_visual_points = 20;
  float voxel_size = 0.5;

  void processMultiCamVIO(
      const std::vector<cv::Mat> &imgs, const std::vector<int> &cam_indices,
      const std::vector<Pose6D> &imu_poses, multimap<double, pointWithVar> &pg,
      const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
      double img_time,
      bool en_cam_backprop // 백프로파게이션 활성화 여부를 인자로 받음
  );

  void computeJacobianAndUpdateEKF(const std::vector<cv::Mat> &imgs,
                                   const std::vector<int> &cam_indices,
                                   const std::vector<Pose6D> &imu_poses,
                                   bool en_cam_backprop);

  void buildJacobianAndResiduals(const cv::Mat &img,
                                 SubSparseMap *current_cam_submap, int level,
                                 VectorXd &z_cam, MatrixXd &H_sub_cam,
                                 VectorXd &R_cam);

  void setCameraTimeOffsets(const std::map<int, double> &time_offsets);
  void compensateExtrinsicsByTimeOffset(const std::vector<Pose6D> &imu_poses,
                                        int cam_idx);
  void backupOriginalExtrinsics();
  void restoreOriginalExtrinsics();

  void setCameraByIndex(int index);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void setExtrinsicParameters(const M3D &R_li, const V3D &P_li,
                              const std::vector<M3D> &R_cl_vec,
                              const std::vector<V3D> &P_cl_vec);
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  StatesGroup *state_prev;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en,
      has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;
  bool dismiss_non_outofbound_pixels_from_ref_patch = false;
  bool en_error_se3_backprop = false;
  bool en_pose_linear_interpolate_backprop = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half,
      border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  double min_cov_pixel;

  // VIO 파라미터들
  float shiTomasiScore_threshold;
  float min_depth_threshold;
  float max_depth_threshold;

  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> *sub_feat_map;
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  Frame *new_frame_;
  cv::Mat img_cp, img_rgb, img_test;
  vector<cv::Mat> img_cps, img_rgbs, img_tests;

  enum CellType { TYPE_MAP = 1, TYPE_POINTCLOUD, TYPE_UNKNOWN };

  VIOManager();
  ~VIOManager();
  void readParameters(ros::NodeHandle &nh);

  // 헬퍼 함수들
  float computeMultiScaleScore(const V2D &pc,
                               const std::vector<cv::Mat> &pyramid_images,
                               double depth);
  bool processVisualPoint(const V3D &pt, const pointWithVar &point_data,
                          const std::vector<cv::Mat> &pyramid_images);

  void updateStateInverse(cv::Mat img, int level);
  // void updateState(cv::Mat img, int level);
  void
  processFrame(cv::Mat &img, multimap<double, pointWithVar> &pg,
               const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
               double img_time);
  void retrieveFromVisualSparseMap(
      cv::Mat img, multimap<double, pointWithVar> &pg,
      const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
      int cam_idx);
  void generateVisualMapPoints(cv::Mat img, multimap<double, pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  // void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam,
                           const Vector2d &px_ref, const Vector3d &f_ref,
                           const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, const int pyramid_level,
                           const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam,
                                     const V2D &px_ref, const V3D &xyz_ref,
                                     const V3D &normal_ref,
                                     const SE3 &T_cur_ref, const int level_ref,
                                     Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref,
                  const Vector2d &px_ref, const int level_ref,
                  const int search_level, const int pyramid_level,
                  const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints(int cam_idx);
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(
      const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(
      const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);

  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;

  // Statistics of visual constraints per pyramid level (per frame)
  // After each processMultiCamVIO call, this will contain the per-level
  // average of total row counts over iterations at that level for the frame.
  std::vector<double> level_avg_visual_points;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_