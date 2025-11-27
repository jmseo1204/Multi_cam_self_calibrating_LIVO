/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

VIOManager::VIOManager() {
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

void VIOManager::readParameters(ros::NodeHandle &nh) {
  nh.param<double>("vio/min_cov_pixel", min_cov_pixel, 100.0);
  nh.param<float>("vio/voxel_size", voxel_size, 0.5);
  nh.param<float>("vio/shiTomasiScore_threshold", shiTomasiScore_threshold,
                  150.0);
  nh.param<float>("vio/min_depth_threshold", min_depth_threshold, 1.5);
  nh.param<float>("vio/max_depth_threshold", max_depth_threshold, 10.0);
  nh.param<int>("vio/min_visual_points", min_visual_points, 20);
  nh.param<bool>("vio/dismiss_non_outofbound_pixels_from_ref_patch",
                 dismiss_non_outofbound_pixels_from_ref_patch, false);
  nh.param<bool>("vio/en_error_se3_backprop", en_error_se3_backprop, false);
  nh.param<bool>("vio/en_dynamic_pixel_var", en_dynamic_pixel_var, false);
  nh.param<bool>("vio/en_pose_linear_interpolate_backprop",
                 en_pose_linear_interpolate_backprop, false);
  nh.param<bool>("common/en_pixelwise_var", en_pixelwise_var, false);

  ROS_INFO("VIO Parameters loaded - shiTomasiScore_threshold: %.1f, "
           "min_depth_threshold: %.1f, max_depth_threshold: %.1f",
           shiTomasiScore_threshold, min_depth_threshold, max_depth_threshold);
}

float VIOManager::computeMultiScaleScore(
    const V2D &pc, const std::vector<cv::Mat> &pyramid_images, double depth) {
  float multi_scale_score = 0.f;
  V2D pc_level = pc;
  const int PYRAMID_LEVELS = pyramid_images.size();

  for (int level = 0; level < PYRAMID_LEVELS; ++level) {
    const cv::Mat &current_img = pyramid_images[level];

    if (pc_level.x() >= 0 && pc_level.x() < current_img.cols &&
        pc_level.y() >= 0 && pc_level.y() < current_img.rows) {

      float score_at_level =
          vk::shiTomasiScore(current_img, pc_level.x(), pc_level.y());

      double huristic_reference_depth = 3.0;
      float weight =
          std::pow(huristic_reference_depth / depth, level) /
          ((std::pow(huristic_reference_depth / depth, PYRAMID_LEVELS) - 1) /
           (huristic_reference_depth / depth - 1));
      multi_scale_score += weight * score_at_level;
    }

    pc_level /= 2.0;
  }

  return multi_scale_score;
}

bool VIOManager::processVisualPoint(
    const V3D &pt, const pointWithVar &point_data,
    const std::vector<cv::Mat> &pyramid_images) {
  V2D pc(new_frame_->w2c(pt));

  if (!new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
    return false;
  }

  V3D dir(new_frame_->T_f_w_ * pt);

  double depth = dir[2];

  if (depth < min_depth_threshold || depth > max_depth_threshold) {
    return false;
  }

  int index = static_cast<int>(pc[1] / grid_size) * grid_n_width +
              static_cast<int>(pc[0] / grid_size);

  if (grid_num[index] == TYPE_MAP) {
    return false;
  }

  float multi_scale_score = computeMultiScaleScore(pc, pyramid_images, depth);

  if (multi_scale_score < shiTomasiScore_threshold) {
    return false;
  }

  if (multi_scale_score > scan_value[index]) {
    scan_value[index] = multi_scale_score;
    append_voxel_points[index] = point_data;
    grid_num[index] = TYPE_POINTCLOUD;
    return true;
  }

  return false;
}

VIOManager::~VIOManager() {
  delete visual_submap;
  for (auto &pair : warp_map)
    delete pair.second;
  warp_map.clear();
  for (auto &pair : feat_map)
    delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot) {
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R,
                                           vector<double> &P) {
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::setExtrinsicParameters(const M3D &rot, const V3D &transl,
                                        const std::vector<M3D> &R_cl_vec_in,
                                        const std::vector<V3D> &P_cl_vec_in) {
  // 1. T_li 저장
  Rli = rot.transpose();
  Pli = -rot.transpose() * transl;

  // 2. T_cl 벡터 저장
  m_R_c_l_vec = R_cl_vec_in;
  m_P_c_l_vec = P_cl_vec_in;

  // 4. 각 카메라에 대한 T_ci를 미리 계산하여 저장
  size_t num_cams = m_cameras.size();
  m_R_c_i_vec.resize(num_cams);
  m_P_c_i_vec.resize(num_cams);

  for (size_t i = 0; i < num_cams; ++i) {
    // R_ci = R_cl * R_li
    m_R_c_i_vec[i] = m_R_c_l_vec[i] * Rli;

    // P_ci = R_cl * P_li + P_cl
    m_P_c_i_vec[i] = m_R_c_l_vec[i] * Pli + m_P_c_l_vec[i];
  }

  ROS_INFO("VIOManager: All extrinsic parameters have been initialized and "
           "T_ci calculated.");
}

void VIOManager::initializeVIO() {
  visual_submaps.resize(m_cameras.size());
  for (int i = 0; i < m_cameras.size(); i++)
    visual_submaps[i] = new SubSparseMap;

  new_frame_vec.resize(m_cameras.size());
  for (int i = 0; i < m_cameras.size(); i++) {
    new_frame_vec[i] = nullptr;
  }

  setCameraByIndex(0);

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = m_cameras[0]->width();
  height = m_cameras[0]->height();

  printf("width: %d, height: %d, scale: %f\n", width, height,
         image_resize_factor);
  if (grid_size > 10) {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  } else {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if (raycast_en) {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n",
           grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++) {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++) {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height ||
            grid_col == grid_n_width)
          border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step) {
          V3D xyz;
          xyz = m_cameras[0]->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]:
    // %d\n", rays_with_sample_points.size(),
    // rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(),
    // rays_with_sample_points[0].size()); for (const auto & it :
    // rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if (colmap_output_en) {
    pinhole_cam = dynamic_cast<vk::PinholeCamera *>(m_cameras[0]);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap
        << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " " << std::fixed
                << std::setprecision(6) // 控制浮点数精度为10位
                << fx << " " << fy << " " << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map->clear();
}

void VIOManager::resetGrid() {
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
// sub_map_ray.clear();
// sub_map_ray_fov.clear();
// visual_sub_map_cur.clear();
// visual_converged_point.clear();
// map_cur_frame.clear();
// sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(V3D p, MD(2, 3) & J) {
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy * z_inv;
  J(1, 2) = -fy * y * z_inv_2;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp,
                               int level) {
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++) {
    uint8_t *img_ptr = (uint8_t *)img.data +
                       (v_ref_i - patch_size_half * scale + x * scale) * width +
                       (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale) {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
          w_ref_bl * img_ptr[scale * width] +
          w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new) {
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  // double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++) {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                          (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end()) {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  } else {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(
    const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref,
    const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref,
    Matrix2d &A_cur_ref) {
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() *
      (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() -
       t * normal_ref.transpose());
  // Compute affine warp matrix A_ref_cur using homography projection
  // const int kHalfPatchSize = 4;
  const int kHalfPatchSize = patch_size_half;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) *
                                          (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) *
                                          (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(
    const vk::AbstractCamera &cam, const Vector2d &px_ref,
    const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
    const int level_ref, const int pyramid_level, const int halfpatch_size,
    Matrix2d &A_cur_ref) {
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(
      cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) *
                                 (1 << pyramid_level)));
  Vector3d xyz_dv_ref(
      cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) *
                                 (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref,
                            const Vector2d &px_ref, const int level_ref,
                            const int search_level, const int pyramid_level,
                            const int halfpatch_size, float *patch) {
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0))) {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 ||
          px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = -1;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] =
            (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref,
                                   const int max_level) {
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level) {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch,
                                int patch_size) {
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++) {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::buildJacobianAndResiduals(const cv::Mat &img,
                                           SubSparseMap *current_cam_submap,
                                           int level, VectorXd &z_cam,
                                           MatrixXd &H_sub_cam,
                                           VectorXd &R_cam) {

  // std::cout << "buildJacobianAndResiduals" << std::endl;

  int num_points = current_cam_submap->voxel_points.size();
  if (num_points == 0) {
    z_cam.resize(0);
    H_sub_cam.resize(0, 7);
    R_cam.resize(0);
    return;
  }

  const int H_DIM = num_points * patch_size_total;
  z_cam.resize(H_DIM);
  H_sub_cam.resize(H_DIM, 7);
  R_cam.resize(H_DIM);

  // 현재 카메라 파라미터(Rci, Pci, Jdphi_dR 등)는 이 함수가 호출되기 전에
  // 외부에서 이미 설정되었다고 가정합니다.
  M3D Rwi(state->rot_end);
  V3D Pwi(state->pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  Jdp_dt = Rci * Rwi.transpose();

#pragma omp parallel for
  for (int i = 0; i < num_points; i++) {
    MD(1, 2) Jimg;
    MD(2, 3) Jdpi;
    MD(1, 3) Jdphi, Jdp, JdR, Jdt;

    VisualPoint *pt = current_cam_submap->voxel_points[i];

    V3D pf = Rcw * pt->pos_ + Pcw;
    V2D pc = cam->world2cam(pf);

    computeProjectionJacobian(pf, Jdpi);
    Vector3d p_diff = pf + Rcw * Pwi - Pci;
    M3D p_hat;
    p_hat << SKEW_SYM_MATRX(p_diff); // p_c - t_cw

    int search_level = current_cam_submap->search_levels[i];
    int pyramid_level = level + search_level;
    int scale = (1 << pyramid_level);
    float inv_scale = 1.0f / scale;

    float u_ref = pc[0];
    float v_ref = pc[1];
    int u_ref_i = floorf(pc[0] / scale) * scale;
    int v_ref_i = floorf(pc[1] / scale) * scale;
    float subpix_u_ref = (u_ref - u_ref_i) / scale;
    float subpix_v_ref = (v_ref - v_ref_i) / scale;
    float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    float w_ref_br = subpix_u_ref * subpix_v_ref;

    const std::vector<float> &P = current_cam_submap->warp_patch[i];
    const std::vector<float> &pixel_variance = current_cam_submap->pixel_var[i];
    double inv_ref_expo = current_cam_submap->inv_expo_list[i];

    for (int x = 0; x < patch_size; x++) {
      uint8_t *img_ptr =
          (uint8_t *)img.data +
          (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i -
          patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale) {

        int row_idx = i * patch_size_total + x * patch_size + y;

        // if (dismiss_non_outofbound_pixels_from_ref_patch) {
        //   if (P[patch_size_total * level + x * patch_size + y] == 0.0) {
        //     z_cam(row_idx) = 0.0;
        //     H_sub_cam.row(row_idx).setZero();
        //     R_cam(row_idx) = 1; // 기본값 사용
        //     continue;
        //   }
        // }

        float du =
            0.5f * ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
                     w_ref_bl * img_ptr[scale * width + scale] +
                     w_ref_br * img_ptr[scale * width + scale * 2]) -
                    (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
                     w_ref_bl * img_ptr[scale * width - scale] +
                     w_ref_br * img_ptr[scale * width]));
        float dv = 0.5f * ((w_ref_tl * img_ptr[scale * width] +
                            w_ref_tr * img_ptr[scale + scale * width] +
                            w_ref_bl * img_ptr[width * scale * 2] +
                            w_ref_br * img_ptr[width * scale * 2 + scale]) -
                           (w_ref_tl * img_ptr[-scale * width] +
                            w_ref_tr * img_ptr[-scale * width + scale] +
                            w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * state->inv_expo_time * inv_scale;

        Jdphi = Jimg * Jdpi * p_hat;
        Jdp = -Jimg * Jdpi;
        // JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
        Vector3d Rcw_Pwi = Pci - Pcw;
        M3D Rcw_Pwi_hat;
        Rcw_Pwi_hat << SKEW_SYM_MATRX(Rcw_Pwi); // p_c - t_cw
        // p_c = R_cw*p_w + t_cw
        // we need to get d_p_c/d_R_wi
        JdR = Jdphi * Jdphi_dR +
              Jdp * Rcw_Pwi_hat * Jdphi_dR; // delta_R_cw + delta_t_cw
        Jdt = Jdp * Jdp_dt;

        double cur_value = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
                           w_ref_bl * img_ptr[scale * width] +
                           w_ref_br * img_ptr[scale * width + scale];
        double res =
            state->inv_expo_time * cur_value -
            inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

        z_cam(row_idx) = res;

        // pixel_var을 사용하여 R_cam 설정

        if (en_dynamic_pixel_var) {
          R_cam(row_idx) =
              pixel_variance[patch_size_total * level + x * patch_size + y];
        } else {
          R_cam(row_idx) = img_point_cov;
        }

        if (exposure_estimate_en) {
          H_sub_cam.block<1, 7>(row_idx, 0) << JdR, Jdt, cur_value;
        } else {
          H_sub_cam.block<1, 6>(row_idx, 0) << JdR, Jdt;
        }
      }
    }
  }
}

void VIOManager::computeJacobianAndUpdateEKF(
    const std::vector<cv::Mat> &imgs, const std::vector<int> &cam_indices,
    const std::vector<Pose6D> &imu_poses, bool en_cam_backprop) {

  // std::cout << "computeJacobianAndUpdateEKF" << std::endl;
  compute_jacobian_time = 0.0;
  update_ekf_time = 0.0;

  // 통계 수집 준비: 레벨별 total_rows 평균 계산용 누적기
  std::vector<double> level_sum_total_rows(
      static_cast<size_t>(patch_pyrimid_level), 0.0);
  std::vector<int> level_iteration_counts(
      static_cast<size_t>(patch_pyrimid_level), 0);

  // 패치 피라미드 레벨에 대한 Coarse-to-Fine 루프
  for (int level = patch_pyrimid_level - 1; level >= 0; level--) {

    StatesGroup old_state = (*state);
    bool EKF_end = false;
    float last_error = std::numeric_limits<float>::max();

    for (int iteration = 0; iteration < max_iterations; iteration++) {
      double t1_jacobian = omp_get_wtime();

      std::vector<MatrixXd> H_list;
      std::vector<VectorXd> z_list;
      std::vector<VectorXd> R_list;
      int total_rows = 0;

      // 1. 각 카메라에 대해 H와 z를 현재 state 기준으로 계산하고 수집
      for (size_t i = 0; i < imgs.size(); ++i) {
        int cam_idx = cam_indices[i];
        const cv::Mat &img = imgs[i];

        setCameraByIndex(cam_idx);

        // b) H, z, R 계산
        VectorXd z_cam;
        MatrixXd H_sub_cam;
        VectorXd R_cam;
        // 해당 카메라의 특징점 리스트(visual_submaps[cam_idx])를 전달
        buildJacobianAndResiduals(img, visual_submaps[cam_idx], level, z_cam,
                                  H_sub_cam, R_cam);

        if (H_sub_cam.rows() > 0) {
          H_list.push_back(H_sub_cam);
          z_list.push_back(z_cam);
          R_list.push_back(R_cam);
          total_rows += H_sub_cam.rows();
        }
      }

      // int min_visual_points = 20;

      if (total_rows / patch_size_total < min_visual_points) {
        if (iteration == 0)
          total_points = 0; // 첫 반복에 포인트가 없으면 0으로 설정
        EKF_end = true;
        break;
      }

      if (iteration == 0)
        total_points = total_rows / patch_size_total;

      // 레벨별 누적 (레벨 인덱스는 0..patch_pyrimid_level-1과 일치시키기 위해
      // level 자체 사용)
      level_sum_total_rows[static_cast<size_t>(level)] +=
          static_cast<double>(total_rows);
      level_iteration_counts[static_cast<size_t>(level)] += 1;

      // 2. 모든 H, z, R을 하나의 큰 행렬/벡터로 결합 (Stacking)
      MatrixXd H_all(total_rows, 7);
      VectorXd z_all(total_rows);
      VectorXd R_all(total_rows);
      int current_row = 0;
      for (size_t i = 0; i < H_list.size(); ++i) {
        H_all.block(current_row, 0, H_list[i].rows(), H_list[i].cols()) =
            H_list[i];
        z_all.segment(current_row, z_list[i].rows()) = z_list[i];
        R_all.segment(current_row, R_list[i].rows()) = R_list[i];
        current_row += H_list[i].rows();
      }

      double t2_jacobian = omp_get_wtime();
      compute_jacobian_time += t2_jacobian - t1_jacobian;

      float error = z_all.squaredNorm() / total_rows;

      if (error < last_error) {
        old_state = (*state);
        last_error = error;

        // 3. 결합된 H, z, R을 사용하여 단일 최적화 수행
        auto &&H_sub_T = H_all.transpose();
        H_T_H.setZero();
        G.setZero();

        // R_all을 사용하여 가중치가 적용된 H_T_H 계산
        // H_T_H = H^T * R^(-1) * H (R은 대각행렬이므로 효율적으로 계산)
        // std::cout << "mean of R: " << R_all.sum() / R_all.size() <<
        // std::endl;

        // R_all의 분위수 계산 및 출력
        // VectorXd R_sorted = R_all;
        // std::sort(R_sorted.data(), R_sorted.data() + R_sorted.size());
        // int n = R_sorted.size();

        // double Q0 = R_sorted(0);                   // 최솟값
        // double Q1 = R_sorted(int((n - 1) * 0.25)); // 25% 분위수
        // double Q2 = R_sorted(int((n - 1) * 0.5));  // 50% 분위수 (중앙값)
        // double Q3 = R_sorted(int((n - 1) * 0.75)); // 75% 분위수
        // double Q4 = R_sorted(n - 1);               // 최댓값

        // std::cout << "R_all quantiles - Q0: " << Q0 << ", Q1: " << Q1
        //           << ", Q2: " << Q2 << ", Q3: " << Q3 << ", Q4: " << Q4
        //           << std::endl;

        // 현재 state의 rotation과 translation covariance 고유값 출력
        // Eigen::Matrix3d rot_cov = state->cov.block<3, 3>(0, 0);
        // Eigen::Matrix3d trans_cov = state->cov.block<3, 3>(3, 3);

        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(rot_cov);
        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans(trans_cov);

        // Eigen::Vector3d rot_eigs = es_rot.eigenvalues();
        // Eigen::Vector3d trans_eigs = es_trans.eigenvalues();

        // // 내림차순으로 정렬
        // std::sort(rot_eigs.data(), rot_eigs.data() + 3,
        // std::greater<double>()); std::sort(trans_eigs.data(),
        // trans_eigs.data() + 3,
        //           std::greater<double>());

        // std::cout << "Rotation cov eigenvalues (desc): " << rot_eigs(0) << ",
        // "
        //           << rot_eigs(1) << ", " << rot_eigs(2) << std::endl;
        // std::cout << "Translation cov eigenvalues (desc): " << trans_eigs(0)
        //           << ", " << trans_eigs(1) << ", " << trans_eigs(2)
        //           << std::endl;

        for (int i = 0; i < total_rows; ++i) {
          // double inv_R_i = 1.0 / std::max(min_cov_pixel, R_all(i));
          double inv_R_i = 1.0 / R_all(i);
          H_T_H.block<7, 7>(0, 0) +=
              inv_R_i * H_all.row(i).transpose() * H_all.row(i);
        }

        MD(DIM_STATE, DIM_STATE) &&K_1 =
            (H_T_H + (state->cov).inverse()).inverse();

        last_cam_solutions.clear();
        last_cam_solutions.reserve(H_list.size());
        last_cam_rot_std.clear();
        last_cam_trans_std.clear();
        last_cam_rot_std.reserve(H_list.size());
        last_cam_trans_std.reserve(H_list.size());

        std::vector<double> all_rot_mags;
        std::vector<double> all_trans_mags;

        for (size_t i = 0; i < H_list.size(); ++i) {
          const int rows_i = static_cast<int>(H_list[i].rows());
          if (rows_i <= 0) {
            last_cam_solutions.emplace_back(MD(DIM_STATE, 1)::Zero());
            last_cam_rot_std.push_back(0.0);
            last_cam_trans_std.push_back(0.0);
            continue;
          }
          VectorXd R_inv_z_i(rows_i);
          for (int r = 0; r < rows_i; ++r) {
            // R is variance per row; invert to get weight
            R_inv_z_i(r) = z_list[i](r) / R_list[i](r);
          }
          VectorXd htz_i = H_list[i].transpose() * R_inv_z_i; // 7x1
          // MD(DIM_STATE, 1) sol_i = -K_1.block<DIM_STATE, 7>(0, 0) * htz_i;

          Matrix<double, 7, 7> HtRH_i = Matrix<double, 7, 7>::Zero();
          for (int r = 0; r < rows_i; ++r) {
            double w = 1.0 / R_list[i](r);
            HtRH_i += w * H_list[i].row(r).transpose() * H_list[i].row(r);
          }
          // Per-camera K_i: embed 7x7 inverse into DIM_STATE x 7 block (others
          // zero)
          Matrix<double, 7, 7> Ki7 =
              (HtRH_i + (state->cov).inverse().block<7, 7>(0, 0)).inverse();
          MD(DIM_STATE, 7) K_i = MD(DIM_STATE, 7)::Zero();
          K_i.block<7, 7>(0, 0) = Ki7;
          MD(DIM_STATE, 1) sol_i = -K_i * htz_i;
          last_cam_solutions.push_back(sol_i);

          // Calculate stats for this camera
          std::vector<double> cam_rot_mags;
          std::vector<double> cam_trans_mags;
          cam_rot_mags.reserve(rows_i);
          cam_trans_mags.reserve(rows_i);

          for (int r = 0; r < rows_i; ++r) {
            double w = 1.0 / R_list[i](r);
            double z_val = z_list[i](r);
            VectorXd u_ir = H_list[i].row(r).transpose() * (z_val * w);
            MD(DIM_STATE, 1) dx_ir = -K_i * u_ir;

            double rot_mag = dx_ir.segment<3>(0).norm();
            double trans_mag = dx_ir.segment<3>(3).norm();

            if (rot_mag > 1e-8) {
              cam_rot_mags.push_back(rot_mag);
              all_rot_mags.push_back(rot_mag);
            }
            if (trans_mag > 1e-8) {
              cam_trans_mags.push_back(trans_mag);
              all_trans_mags.push_back(trans_mag);
            }
          }

          double rot_mean = 0, rot_sq_sum = 0;
          double trans_mean = 0, trans_sq_sum = 0;
          
          if (!cam_rot_mags.empty()) {
            for (double v : cam_rot_mags) {
              rot_mean += v;
              rot_sq_sum += v * v;
            }
            rot_mean /= cam_rot_mags.size();
            double rot_var = (rot_sq_sum / cam_rot_mags.size()) - (rot_mean * rot_mean);
            last_cam_rot_std.push_back(std::sqrt(std::max(0.0, rot_var)));
          } else {
             last_cam_rot_std.push_back(0.0);
          }

          if (!cam_trans_mags.empty()) {
            for (double v : cam_trans_mags) {
              trans_mean += v;
              trans_sq_sum += v * v;
            }
            trans_mean /= cam_trans_mags.size();
            double trans_var = (trans_sq_sum / cam_trans_mags.size()) - (trans_mean * trans_mean);
            last_cam_trans_std.push_back(std::sqrt(std::max(0.0, trans_var)));
          } else {
            last_cam_trans_std.push_back(0.0);
          }
        }

        // Compute total std dev
        if (!all_rot_mags.empty()) {
          double rot_mean = 0, rot_sq_sum = 0;
          size_t N = all_rot_mags.size();
          for (double v : all_rot_mags) {
            rot_mean += v;
            rot_sq_sum += v * v;
          }
          rot_mean /= N;
          double rot_var = (rot_sq_sum / N) - (rot_mean * rot_mean);
          last_total_rot_std = std::sqrt(std::max(0.0, rot_var));
        } else {
          last_total_rot_std = 0.0;
        }

        if (!all_trans_mags.empty()) {
          double trans_mean = 0, trans_sq_sum = 0;
          size_t N = all_trans_mags.size();
          for (double v : all_trans_mags) {
            trans_mean += v;
            trans_sq_sum += v * v;
          }
          trans_mean /= N;
          double trans_var = (trans_sq_sum / N) - (trans_mean * trans_mean);
          last_total_trans_std = std::sqrt(std::max(0.0, trans_var));
        } else {
          last_total_trans_std = 0.0;
        }

        // R^(-1) * z 계산
        VectorXd R_inv_z(total_rows);
        for (int i = 0; i < total_rows; ++i) {
          // double inv_R_i = 1.0 / std::max(1e-2, R_all(i));
          double inv_R_i = 1.0 / R_all(i);
          R_inv_z(i) = z_all(i) * inv_R_i;
        }
        auto &&HTz = H_sub_T * R_inv_z;
        auto vec = (*state_propagat) - (*state);
        G.block<DIM_STATE, 7>(0, 0) =
            K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
        MD(DIM_STATE, 1)
        solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec -
                   G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);
        // solution +=
        // K_1 * (*state_propagat).cov.inverse() * vec; // prior
        // regularization

        // 이거 추가하면 G를 P_IMU로
        // 구성해야하는데, 애초부터 식이 잘못됨. 그리고 IMU 너무 부정확해서
        // prior이 무의미

        // std::cout << "state update" << std::endl;

        (*state) += solution;

        auto &&rot_add = solution.block<3, 1>(0, 0);
        auto &&t_add = solution.block<3, 1>(3, 0);
        if ((rot_add.norm() * 57.3f < 0.01f) &&
            (t_add.norm() * 100.0f < 0.015f)) {
          EKF_end = true;
        }
      } else {
        (*state) = old_state;
        EKF_end = true;
      }

      update_ekf_time += omp_get_wtime() - t2_jacobian;

      if (EKF_end)
        break;
    }
  }

  std::cout << "level update" << std::endl;
  // 레벨별 평균을 멤버에 기록 (프레임 단위)
  level_avg_visual_points.clear();
  level_avg_visual_points.resize(static_cast<size_t>(patch_pyrimid_level), 0.0);
  for (int l = 0; l < patch_pyrimid_level; ++l) {
    int cnt = level_iteration_counts[static_cast<size_t>(l)];
    level_avg_visual_points[static_cast<size_t>(l)] =
        (cnt > 0) ? (level_sum_total_rows[static_cast<size_t>(l)] /
                     static_cast<double>(cnt)) /
                        patch_size_total
                  : 0.0;
  }

  // 최종 공분산 업데이트 및 상태 반영
  state->cov -= G * state->cov;
}

// void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img) {
//   if (total_points == 0)
//     return;

//   compute_jacobian_time = update_ekf_time = 0.0;

//   for (int level = patch_pyrimid_level - 1; level >= 0; level--) {
//     if (inverse_composition_en) {
//       has_ref_patch_cache = false;
//       updateStateInverse(img, level);
//     } else
//       updateState(img, level);
//   }
//   state->cov -= G * state->cov;
//   updateFrameState(*state);
// }

void VIOManager::retrieveFromVisualSparseMap(
    cv::Mat img, multimap<double, pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
    int cam_idx) {

  if (feat_map.size() <= 0)
    return;

  // std::cout << "retrieveFromVisualSparseMap" << std::endl;

  double ts0 = omp_get_wtime();

  // pg_down->reserve(feat_map.size());
  // downSizeFilter.setInputCloud(pg);
  // downSizeFilter.filter(*pg_down);

  // resetRvizDisplay();
  visual_submap = visual_submaps[cam_idx];
  visual_submap->reset();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map->clear();

  // float voxel_size = 0.5;

  if (!normal_en)
    warp_map.clear();

  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  // float it[height * width] = {0.0};

  // double t_insert, t_depth, t_position;
  // t_insert=t_depth=t_position=0;

  int loc_xyz[3];

  // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
  // double ts1 = omp_get_wtime();

  // printf("pg size: %zu \n", pg.size());

  for (const auto &pair : pg) {
    // double t0 = omp_get_wtime();

    V3D pt_w = pair.second.point_w;

    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // t_position += omp_get_wtime()-t0;
    // double t1 = omp_get_wtime();

    auto iter = sub_feat_map->find(position);
    if (iter == sub_feat_map->end()) {
      (*sub_feat_map)[position] = 0;
    } else {
      iter->second = 0;
    }

    // t_insert += omp_get_wtime()-t1;
    // double t2 = omp_get_wtime();

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0) {
      V2D px;
      // px[0] = fx * pt_c[0]/pt_c[2] + cx;
      // px[1] = fy * pt_c[1]/pt_c[2]+ cy;
      px = new_frame_->cam_->world2cam(pt_c);

      if (new_frame_->cam_->isInFrame(px.cast<int>(), border)) {
        // cv::circle(img_cp, cv::Point2f(px[0], px[1]), 3, cv::Scalar(0, 0,
        // 255), -1, 8);
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width * row + col] = depth;
      }
    }
    // t_depth += omp_get_wtime()-t2;
  }

  // imshow("depth_img", depth_img);
  // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
  // printf("A11. calculate pt position: %.6lf \n", t_position);
  // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
  // printf("A13. generate depth map: %.6lf \n", t_depth);
  // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);

  // double t1 = omp_get_wtime();
  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : (*sub_feat_map)) {
    VOXEL_LOCATION position = iter.first;

    // double t4 = omp_get_wtime();
    auto corre_voxel = feat_map.find(position);
    // double t5 = omp_get_wtime();

    if (corre_voxel != feat_map.end()) {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points =
          corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++) {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr)
          continue;
        if (pt->obs_.size() == 0)
          continue;

        V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0)
          continue;

        if (dir[2] > max_depth_threshold || dir[2] < min_depth_threshold)
          continue;
        // dir.normalize();
        // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree  0.17 80
        // degree 0.08 85 degree

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
          // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 255,
          // 255), -1, 8);
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width +
                      static_cast<int>(pc[0] / grid_size);
          grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist[index]) {
            map_dist[index] = cur_dist;
            retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) {
        DeleteKeyList.push_back(position);
      }
    }
  }

  // RayCasting Module
  if (raycast_en) {
    for (int i = 0; i < length; i++) {
      if (grid_num[i] == TYPE_MAP || border_flag[i] == 1)
        continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points[i]) {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
          }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map->find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map->end())
          break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end()) {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points =
              corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0)
            continue;

          for (int j = 0; j < voxel_num; j++) {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr)
              continue;
            if (pt->obs_.size() == 0)
              continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0)
              continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17
            // 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border)) {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3,
              // cv::Scalar(255, 255, 0), -1, 8); sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true; // 90도 제한
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width +
                          static_cast<int>(pc[0] / grid_size);
              grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist[index]) {
                map_dist[index] = cur_dist;
                retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov)
            (*sub_feat_map)[sample_pos] = 0;
          break;
        } else {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end()) {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_) {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList) {
    sub_feat_map->erase(key);
  }

  // double t2 = omp_get_wtime();

  // cout<<"B. feat_map.find: "<<t2-t1<<endl;

  // double t_2, t_3, t_4, t_5;
  // t_2=t_3=t_4=t_5=0;

  for (int i = 0; i < length; i++) {
    if (grid_num[i] == TYPE_MAP) {
      // double t_1 = omp_get_wtime();

      VisualPoint *pt = retrieve_voxel_points[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255),
      // -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++) {
        for (int v = -patch_size_half; v <= patch_size_half; v++) {
          if (u == 0 && v == 0)
            continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.)
            continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5) {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous)
          break;
      }
      if (depth_continous)
        continue;

      // t_2 += omp_get_wtime() - t_1;

      // t_1 = omp_get_wtime();
      if (!pt->is_normal_initialized_)
        continue;

      // 현재 프레임의 월드에 대한 회전 행렬
      const M3D &current_R_w_c =
          new_frame_->T_f_w_.rotation_matrix()
              .transpose(); // T_f_w_는 T_c_w, 따라서 transpose해야 R_w_c

      // 각 패치와 현재 프레임 간의 회전 차이를 저장할 벡터
      std::vector<std::pair<double, Feature *>> candidate_features;

      for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it) {
        Feature *ftr = *it;
        const M3D &past_R_w_c = ftr->T_f_w_.rotation_matrix().transpose();

        // acos((tr(R1^T * R2) - 1) / 2)가 더 정확하지만, norm이 더 빠름

        double cos_sim =
            0.5 * ((current_R_w_c.transpose() * past_R_w_c).trace() - 1);
        candidate_features.push_back({cos_sim, ftr});
      }

      // 회전 차이가 작은 순서대로 정렬
      std::sort(candidate_features.begin(), candidate_features.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

      // 최대 10개 또는 obs_ 사이즈 중 작은 값을 선택
      int num_patches_to_use = std::min((int)candidate_features.size(), 10);
      if (num_patches_to_use == 0) {
        continue; // 사용할 패치가 없으면 이 포인트는 건너뜀
      }

      // 2. 선택된 여러 패치를 Warping하여 가중평균 및 분산 계산 (최적화된 단일
      // 패스)

      std::vector<float> avg_warped_patch(
          patch_size_total * patch_pyrimid_level, 0.0f);
      std::vector<float> pixel_var(
          patch_size_total * patch_pyrimid_level,
          1000.0 * min_cov_pixel); // 분산 벡터, 기본값 MAX var
      std::vector<int> pixel_count(patch_size_total * patch_pyrimid_level,
                                   0); // 픽셀별 count 벡터
      std::vector<float> temp_warped_patch(warp_len);
      double weighted_inv_expo = 0.0;
      double total_weight = 0.0;

      // 각 패치의 warping 결과를 저장할 벡터
      std::vector<std::vector<float>> warped_patches;
      std::vector<double> patch_weights;
      std::vector<double> patch_inv_expos;

      // 첫 번째 패스: 모든 패치를 warping하고 결과 저장
      for (int p_idx = 0; p_idx < num_patches_to_use; ++p_idx) {
        Feature *ref_ftr = candidate_features[p_idx].second;
        double cos_sim = candidate_features[p_idx].first;

        // 가중치 계산: cos(cos_sim) 사용
        double weight = cos_sim;
        if (weight < 0)
          weight = 0; // 음수 가중치 방지

        // 각 후보 패치에 대한 아핀 변환 행렬 계산 (기존 로직과 동일)
        Matrix2d A_cur_ref_zero;
        int search_level;

        // normal_en 플래그에 따라 아핀 행렬 계산 (기존과 동일)
        if (normal_en) {
          V3D norm_vec =
              (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
          V3D pf(ref_ftr->T_f_w_ * pt->pos_);
          SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
          getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec,
                                        T_cur_ref, 0, A_cur_ref_zero);
          search_level = getBestSearchLevel(A_cur_ref_zero, 2);
        } else {
          auto iter_warp = warp_map.find(ref_ftr->id_);
          if (iter_warp != warp_map.end()) {
            search_level = iter_warp->second->search_level;
            A_cur_ref_zero = iter_warp->second->A_cur_ref;
          } else {
            getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_,
                                (ref_ftr->pos() - pt->pos_).norm(),
                                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                                ref_ftr->level_, 0, patch_size_half,
                                A_cur_ref_zero);

            search_level = getBestSearchLevel(A_cur_ref_zero, 2);

            Warp *ot = new Warp(search_level, A_cur_ref_zero);
            warp_map[ref_ftr->id_] = ot;
          }
        }

        // 각 패치를 현재 프레임으로 warping
        for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1;
             pyramid_level++) {
          warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_,
                     ref_ftr->level_, search_level, pyramid_level,
                     patch_size_half, temp_warped_patch.data());
        }

        // warping 결과 저장
        warped_patches.push_back(std::vector<float>(temp_warped_patch.begin(),
                                                    temp_warped_patch.end()));
        patch_weights.push_back(weight);
        patch_inv_expos.push_back(ref_ftr->inv_expo_time_);

        total_weight += weight;
        weighted_inv_expo += ref_ftr->inv_expo_time_ * weight;
      }

      if (total_weight == 0.0)
        continue;

      // 두 번째 패스: 가중평균과 가중분산을 동시에 계산
      for (size_t k = 0; k < avg_warped_patch.size(); ++k) {
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        int valid_count = 0;

        // 가중평균 계산
        for (int p_idx = 0; p_idx < num_patches_to_use; ++p_idx) {
          if (warped_patches[p_idx][k] > 1e-6) {
            float warped_value =
                warped_patches[p_idx][k] * patch_inv_expos[p_idx];
            weighted_sum += warped_value * patch_weights[p_idx];
            weight_sum += patch_weights[p_idx];
            valid_count++;
          }
        }

        if (valid_count > 0) {
          avg_warped_patch[k] = weighted_sum / weight_sum;
          pixel_count[k] = valid_count;

          // 가중분산 계산 (Welford's online algorithm 변형)
          double weighted_variance = 0.0;
          for (int p_idx = 0; p_idx < num_patches_to_use; ++p_idx) {
            if (warped_patches[p_idx][k] > 1e-6) {
              float warped_value =
                  warped_patches[p_idx][k] * patch_inv_expos[p_idx];
              double diff = warped_value - avg_warped_patch[k];
              weighted_variance += patch_weights[p_idx] * diff * diff;
            }
          }

          if (en_pixelwise_var) {
            // 픽셀별 분산 적용
            if (valid_count == 0) {
              pixel_var[k] =
                  1000.0 * min_cov_pixel; // var INF -> impact minimized
            } else if (valid_count <= 3) {
              pixel_var[k] = 3.0 * min_cov_pixel;
            } else {
              pixel_var[k] =
                  std::max(weighted_variance / weight_sum * img_point_cov,
                           min_cov_pixel); // 가중분산 정규화
            }
          } else {
            pixel_var[k] = img_point_cov; // var INF -> impact minimized
          }
        }
      }

      double avg_inv_expo = weighted_inv_expo / total_weight;

      // 3. 현재 이미지 패치와 평균 reprojected patch 간의 에러 계산

      getImagePatch(img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++) {
        // 주의: avg_warped_patch는 이미 inv_expo_time이 곱해져 있음
        float residual =
            avg_warped_patch[ind] - state->inv_expo_time * patch_buffer[ind];
        error += residual * residual;
      }

      if (ncc_en) {
        double ncc = calculateNCC(avg_warped_patch.data(), patch_buffer.data(),
                                  patch_size_total);
        if (ncc < ncc_thre) {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total)
        continue;

      // 4. 최종 결과 저장
      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      // search_level, warp_patch, inv_expo_list는 이제 평균 또는 대표값을
      // 저장해야 함
      visual_submap->search_levels.push_back(
          candidate_features[0].second->level_); // 대표로 첫 번째 값 사용
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(avg_warped_patch); // 평균 패치 저장
      visual_submap->pixel_var.push_back(pixel_var);         // 픽셀별 분산 저장
      visual_submap->inv_expo_list.push_back(
          avg_inv_expo); // 평균 노출 시간 저장

      // ================================================================== //
      // ### 수정 로직 끝 ###
      // ================================================================== //

      // t_5 += omp_get_wtime() - t_1;
    }
  }
  total_points = visual_submap->voxel_points.size();
  visual_submaps[cam_idx] = visual_submap;

  // double t3 = omp_get_wtime();
  // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
  // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4
  // "<<t_5<<endl;
  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points);
}

void VIOManager::generateVisualMapPoints(cv::Mat img,
                                         multimap<double, pointWithVar> &pg) {

  if (pg.size() <= 10)
    return;

  if (img.channels() == 3) {
    cv::cvtColor(img, img, CV_BGR2GRAY);
  }
  // double t0 = omp_get_wtime();
  // for (int i = 0; i < pg.size(); i++) {
  for (auto pg_it = pg.begin(); pg_it != pg.end(); ++pg_it) {
    auto pg_i = pg_it->second;
    if (pg_i.normal == V3D(0, 0, 0))
      continue;

    V3D pt = pg_i.point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(
            pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width +
                  static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP) {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value[index]) {
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg_i;
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++) {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));

    if (new_frame_->cam_->isInFrame(
            pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width +
                  static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP) {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value[index]) {
          scan_value[index] = cur_value;
          append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++) {
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new =
          new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) {
        pt_new->normal_ = -pt_var.normal;
      } else {
        pt_new->normal_ = pt_var.normal;
      }

      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

void VIOManager::updateVisualMapPoints(cv::Mat img) {
  if (total_points == 0)
    return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points; i++) {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr)
      continue;
    if (pt->is_converged_) {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;

    float *patch_temp = new float[patch_size_total];
    getImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta =
        (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6)
            ? 0.0
            : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.2)
      add_flag = true; // 0.5 || 0.3

    // Step 3: pixel distance
    Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40)
      add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30) {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag) {
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_,
                                     visual_submap->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map) {
  if (total_points == 0)
    return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_)
      continue;
    if (pt->is_converged_)
      continue;
    if (pt->obs_.size() <= 5)
      continue;
    if (update_flag[i] == 0)
      continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end()) {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_) {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) +
                             plane.normal_(1) * p_w(1) +
                             plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center =
            (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
            (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
            (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_) {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l +=
              plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l)) {
            // V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() *
            // plane.normal_); V3D pf(new_frame_->T_f_w_ * pt->pos_); V3D
            // pf_ref(pt->ref_patch->T_f_w_ * pt->pos_); V3D
            // norm_vec_ref(pt->ref_patch->T_f_w_.rotation_matrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);

            if (pt->previous_normal_.dot(plane.normal_) < 0) {
              pt->normal_ = -plane.normal_;
            } else {
              pt->normal_ = plane.normal_;
            }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10) {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it) {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6) {
        float ref_sum =
            std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_mean = ref_sum / patch_size_total;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme;
           ++itm) {
        if ((*itm)->id_ == ref_patch_temp->id_)
          continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6) {
          float other_sum =
              std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          other_mean = other_sum / patch_size_total;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total; ind++) {
          NCC_up +=
              (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 +=
              (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 +=
              (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max) {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }
  }
}

void VIOManager::projectPatchFromRefToCur(
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map) {
  if (total_points == 0)
    return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_) {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0)
        norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(*cam, ref_ftr->px_, pf, norm_vec, T_cur_ref,
                                    0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3)
        continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++) {
        error_est +=
            (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] -
             state->inv_expo_time * patch_buffer[ind]) *
            (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] -
             state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est =
          "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est =
          "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est,
                  cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40,
                              ref_ftr->px_[1] + 40),
                  cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::putText(ref_cur_combine_temp, cur_est,
                  cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa,
                  cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 0, 255), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_optimize,
                  cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius,
                                ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius,
                                ref_ftr->px_[1] + radius),
                    cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(pc[0] - radius, pc[1] - radius),
                    cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius),
                    cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp,
                 cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]),
                 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1,
                 cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1,
                 cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" +
                      std::to_string(ref_ftr->id_) + "_" + std::to_string(num) +
                      ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y) {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value =
              (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch +
                            pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 ||
              px[1] >= img_cur.rows - 1)
            continue;
          else {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++) {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++) {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min)
            x_min = x_temp;
          if (y_temp < y_min)
            y_min = y_temp;
          if (x_temp > x_max)
            x_max = x_temp;
          if (y_temp > y_max)
            y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++) {
        for (int j = y_min_i; j < y_max_i; j++) {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) &&
              px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) &&
              px_patch[1] < (patch_size / 2 * (1 << search_level))) {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value =
                (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_)
      continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(*cam, ref_ftr->px_, ref_ftr->f_,
                        (ref_ftr->pos() - pt->pos_).norm(),
                        new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3)
      continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value =
            (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch +
                          pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 ||
            px[1] >= img_cur.rows - 1)
          continue;
        else {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png",
              ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png",
              ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level) {
  double t1 = omp_get_wtime();
  if (total_points == 0)
    return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++) {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr)
      continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++) {
      uint8_t *img_ptr =
          (uint8_t *)img.data +
          (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i -
          patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale) {
        float du =
            0.5f * ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
                     w_ref_bl * img_ptr[scale * width + scale] +
                     w_ref_br * img_ptr[scale * width + scale * 2]) -
                    (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
                     w_ref_bl * img_ptr[scale * width - scale] +
                     w_ref_br * img_ptr[scale * width]));
        float dv = 0.5f * ((w_ref_tl * img_ptr[scale * width] +
                            w_ref_tr * img_ptr[scale + scale * width] +
                            w_ref_bl * img_ptr[width * scale * 2] +
                            w_ref_br * img_ptr[width * scale * 2 + scale]) -
                           (w_ref_tl * img_ptr[-scale * width] +
                            w_ref_tr * img_ptr[-scale * width + scale] +
                            w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0)
            << JdR,
            Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(cv::Mat img, int level) {
  if (total_points == 0)
    return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false)
      precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++) {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr)
        continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++) {
        uint8_t *img_ptr =
            (uint8_t *)img.data +
            (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i -
            patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale) {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
                       w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] -
                       P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3)
          J_dR = H_sub_inv.block<1, 3>(
              i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3)
          J_dt = H_sub_inv.block<1, 3>(
              i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0)
              << JdR,
              Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error) {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 =
          (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) =
          K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                      G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) &&
          (t_add.norm() * 100.0f < 0.001f)) {
        EKF_end = true;
      }
    } else {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end)
      break;
  }
}

void VIOManager::setCameraByIndex(int index) {
  if (index < 0 || index >= m_cameras.size()) {
    ROS_ERROR("VIOManager: Invalid camera index provided: %d", index);
    return;
  }

  this->cam = m_cameras[index];
  this->fx = cam->fx();
  this->fy = cam->fy();
  this->cx = cam->cx();
  this->cy = cam->cy();
  this->image_resize_factor = cam->scale();
  this->sub_feat_map = sub_feat_maps[index];
  this->visual_submap = visual_submaps[index];

  this->Rci = m_R_c_i_vec[index];
  this->Pci = m_P_c_i_vec[index];
  this->Rcl = m_R_c_l_vec[index];
  this->Pcl = m_P_c_l_vec[index];

  V3D Pic;
  M3D tmp;
  this->Jdphi_dR = Rci;
  Pic = -Rci.transpose() * Pci;
  tmp << SKEW_SYM_MATRX(Pic);
  this->Jdp_dR = -Rci * tmp;

  this->new_frame_ = new_frame_vec[index];
  updateFrameState(*state);

  // this->cam_idx = index;
}

// In vio.cpp

void VIOManager::processMultiCamVIO(
    const std::vector<cv::Mat> &imgs, const std::vector<int> &cam_indices,
    const std::vector<Pose6D> &imu_poses, multimap<double, pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
    double img_time, bool en_cam_backprop) {
  // 타이머 및 프레임 카운트 변수들
  std::cout << "processMultiCamVIO" << std::endl;

  double t1(0), t2(0), t3(0), t4(0), t5(0), t6(0), t7(0);
  t1 = omp_get_wtime();

  img_cps.clear();
  img_rgbs.clear();
  img_cps.resize(m_cameras.size());
  img_rgbs.resize(m_cameras.size());

  std::vector<cv::Mat> processed_imgs;

  backupOriginalExtrinsics();

  for (size_t i = 0; i < imgs.size(); ++i) {
    int cam_idx = cam_indices[i];
    cv::Mat current_img = imgs[i];

    if (width != current_img.cols || height != current_img.rows) {
      cv::resize(current_img, current_img, cv::Size(width, height), 0, 0,
                 CV_INTER_LINEAR);
    }

    img_cps[cam_idx] = current_img.clone();
    img_rgbs[cam_idx] = current_img.clone();

    if (current_img.channels() == 3) {
      cv::cvtColor(current_img, current_img, CV_BGR2GRAY);
    }
    delete new_frame_vec[cam_idx];
    new_frame_vec[cam_idx] = new Frame(cam, current_img);

    if (en_cam_backprop) {
      compensateExtrinsicsByTimeOffset(imu_poses, cam_idx);
    } else {
      setCameraByIndex(cam_idx);
    }

    resetGrid();

    retrieveFromVisualSparseMap(current_img, pg, feat_map, cam_idx);

    processed_imgs.push_back(current_img);
  }

  t2 = omp_get_wtime();

  computeJacobianAndUpdateEKF(processed_imgs, cam_indices, imu_poses,
                              en_cam_backprop);

  t3 = omp_get_wtime();

  for (size_t i = 0; i < processed_imgs.size(); ++i) {

    int cam_idx = cam_indices[i];
    cv::Mat &img = processed_imgs[i];
    std::cout << "[ VIO ] Processing for camera index: " << cam_idx
              << std::endl;

    setCameraByIndex(cam_idx);

    total_points = visual_submap->voxel_points.size();

    std::cout << "[ VIO ] generateVisualMapPoints" << std::endl;
    // 후속 작업들
    generateVisualMapPoints(img_rgbs[cam_idx], pg);
    t4 = (t4 * i + omp_get_wtime()) / (i + 1);

    std::cout << "[ VIO ] projectPatchFromRefToCur" << std::endl;

    plotTrackedPoints(cam_idx);

    if (plot_flag)
      projectPatchFromRefToCur(feat_map);
    t5 = (t5 * i + omp_get_wtime()) / (i + 1);

    std::cout << "[ VIO ] updateVisualMapPoints" << std::endl;
    updateVisualMapPoints(img);
    t6 = (t6 * i + omp_get_wtime()) / (i + 1);

    std::cout << "[ VIO ] updateReferencePatch" << std::endl;
    updateReferencePatch(feat_map);
    t7 = (t7 * i + omp_get_wtime()) / (i + 1);

    if (colmap_output_en)
      dumpDataForColmap();
  }
  std::cout << "[ VIO ] restoreOriginalExtrinsics" << std::endl;

  restoreOriginalExtrinsics();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count +
              (t7 - t1 - (t5 - t4)) / frame_count;

  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                         "
         "   |\033[0m\n");
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size",
         feat_map.size());
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap",
         t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF",
         t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian",
         compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF",
         update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints",
         t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints",
         t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch",
         t7 - t6);
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time",
         t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time",
         ave_total);
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
}

// void VIOManager::updateState(cv::Mat img, int level) {
//   if (total_points == 0)
//     return;
//   StatesGroup old_state = (*state);

//   VectorXd z;
//   MatrixXd H_sub;
//   bool EKF_end = false;
//   float last_error = std::numeric_limits<float>::max();

//   const int H_DIM = total_points * patch_size_total;
//   z.resize(H_DIM);
//   z.setZero();
//   H_sub.resize(H_DIM, 7);
//   H_sub.setZero();

//   for (int iteration = 0; iteration < max_iterations; iteration++) {
//     double t1 = omp_get_wtime();

//     M3D Rwi(state->rot_end);
//     V3D Pwi(state->pos_end);
//     Rcw = Rci * Rwi.transpose();
//     Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
//     Jdp_dt = Rci * Rwi.transpose();

//     float error = 0.0;
//     int n_meas = 0;
//     // int max_threads = omp_get_max_threads();
//     // int desired_threads = std::min(max_threads, total_points);
//     // omp_set_num_threads(desired_threads);

// #ifdef MP_EN
//     omp_set_num_threads(MP_PROC_NUM);
// #pragma omp parallel for reduction(+ : error, n_meas)
// #endif
//     for (int i = 0; i < total_points; i++) {
//       // printf("thread is %d, i=%d, i address is %p\n",
//       omp_get_thread_num(),
//       // i, &i);
//       MD(1, 2) Jimg;
//       MD(2, 3) Jdpi;
//       MD(1, 3) Jdphi, Jdp, JdR, Jdt;

//       float patch_error = 0.0;
//       int search_level = visual_submap->search_levels[i];
//       int pyramid_level = level + search_level;
//       int scale = (1 << pyramid_level);
//       float inv_scale = 1.0f / scale;

//       VisualPoint *pt = visual_submap->voxel_points[i];

//       if (pt == nullptr)
//         continue;

//       V3D pf = Rcw * pt->pos_ + Pcw;
//       V2D pc = cam->world2cam(pf);

//       computeProjectionJacobian(pf, Jdpi);
//       M3D p_hat;
//       p_hat << SKEW_SYM_MATRX(pf);

//       float u_ref = pc[0];
//       float v_ref = pc[1];
//       int u_ref_i = floorf(pc[0] / scale) * scale;
//       int v_ref_i = floorf(pc[1] / scale) * scale;
//       float subpix_u_ref = (u_ref - u_ref_i) / scale;
//       float subpix_v_ref = (v_ref - v_ref_i) / scale;
//       float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
//       float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
//       float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
//       float w_ref_br = subpix_u_ref * subpix_v_ref;

//       vector<float> P = visual_submap->warp_patch[i];
//       double inv_ref_expo = visual_submap->inv_expo_list[i];
//       // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n",
//       // inv_ref_expo, state->inv_expo_time);

//       for (int x = 0; x < patch_size; x++) {
//         uint8_t *img_ptr =
//             (uint8_t *)img.data +
//             (v_ref_i + x * scale - patch_size_half * scale) * width +
//             u_ref_i
//             - patch_size_half * scale;
//         for (int y = 0; y < patch_size; ++y, img_ptr += scale) {
//           float du =
//               0.5f *
//               ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
//                 w_ref_bl * img_ptr[scale * width + scale] +
//                 w_ref_br * img_ptr[scale * width + scale * 2]) -
//                (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
//                 w_ref_bl * img_ptr[scale * width - scale] +
//                 w_ref_br * img_ptr[scale * width]));
//           float dv =
//               0.5f * ((w_ref_tl * img_ptr[scale * width] +
//                        w_ref_tr * img_ptr[scale + scale * width] +
//                        w_ref_bl * img_ptr[width * scale * 2] +
//                        w_ref_br * img_ptr[width * scale * 2 + scale]) -
//                       (w_ref_tl * img_ptr[-scale * width] +
//                        w_ref_tr * img_ptr[-scale * width + scale] +
//                        w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

//           Jimg << du, dv;
//           Jimg = Jimg * state->inv_expo_time;
//           Jimg = Jimg * inv_scale;
//           Jdphi = Jimg * Jdpi * p_hat;
//           Jdp = -Jimg * Jdpi;
//           JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
//           Jdt = Jdp * Jdp_dt;

//           double cur_value = w_ref_tl * img_ptr[0] + w_ref_tr *
//           img_ptr[scale] +
//                              w_ref_bl * img_ptr[scale * width] +
//                              w_ref_br * img_ptr[scale * width + scale];
//           double res =
//               state->inv_expo_time * cur_value -
//               inv_ref_expo * P[patch_size_total * level + x * patch_size +
//               y];

//           z(i * patch_size_total + x * patch_size + y) = res;

//           patch_error += res * res;
//           n_meas += 1;

//           if (exposure_estimate_en) {
//             H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0)
//                 << JdR,
//                 Jdt, cur_value;
//           } else {
//             H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0)
//                 << JdR,
//                 Jdt;
//           }
//         }
//       }
//       visual_submap->errors[i] = patch_error;
//       error += patch_error;
//     }

//     error = error / n_meas;

//     compute_jacobian_time += omp_get_wtime() - t1;

//     // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
//     // std::cout << "It. " << iteration
//     //           << "\t last_error = " << last_error
//     //           << "\t new_error = " << error
//     //           << std::endl;

//     double t3 = omp_get_wtime();

//     if (error <= last_error) {
//       old_state = (*state);
//       last_error = error;

//       // K = (H.transpose() / img_point_cov * H +
//       // state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
//       auto
//       // vec = (*state_propagat) - (*state); G = K*H;
//       // (*state) += (-K*z + vec - G*vec);

//       auto &&H_sub_T = H_sub.transpose();
//       H_T_H.setZero();
//       G.setZero();
//       H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
//       MD(DIM_STATE, DIM_STATE) &&K_1 =
//           (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
//       auto &&HTz = H_sub_T * z;
//       // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
//       auto vec = (*state_propagat) - (*state);
//       G.block<DIM_STATE, 7>(0, 0) =
//           K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
//       MD(DIM_STATE, 1)
//       solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec -
//                  G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

//       (*state) += solution;
//       auto &&rot_add = solution.block<3, 1>(0, 0);
//       auto &&t_add = solution.block<3, 1>(3, 0);

//       auto &&expo_add = solution.block<1, 1>(6, 0);
//       // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f <
//       // 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
//       if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f <
//       0.001f))
//         EKF_end = true;
//     } else {
//       (*state) = old_state;
//       EKF_end = true;
//     }

//     update_ekf_time += omp_get_wtime() - t3;

//     if (iteration == max_iterations || EKF_end)
//       break;
//   }
//   // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo
//   time!!!!!!!!!!\n");
//   // state->inv_expo_time = 0.0;}
// }

void VIOManager::updateFrameState(StatesGroup state) {
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  if (new_frame_ != nullptr)
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void VIOManager::plotTrackedPoints(int cam_idx) {
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0)
    return;
  // int inlier_count = 0;
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Poaint2f(0, grid_size * i),
  //   cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1,
  //   CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size *
  //   i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  // for (int i = 0; i < img_cp.rows / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(0, grid_size * i),
  //   cv::Point2f(img_cp.cols, grid_size * i), cv::Scalar(255, 255, 255), 1,
  //   CV_AA);
  // }
  // for (int i = 0; i < img_cp.cols / grid_size; i++)
  // {
  //   cv::line(img_cp, cv::Point2f(grid_size * i, 0), cv::Point2f(grid_size *
  //   i, img_cp.rows), cv::Scalar(255, 255, 255), 1, CV_AA);
  // }
  for (int i = 0; i < total_points; i++) {
    VisualPoint *pt = visual_submap->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i]) {
      // inlier_count++;
      cv::circle(img_cps[cam_idx], cv::Point2f(pc[0], pc[1]), 7,
                 cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    } else {
      cv::circle(img_cps[cam_idx], cv::Point2f(pc[0], pc[1]), 7,
                 cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
  // std::string text = std::to_string(inlier_count) + " " +
  // std::to_string(total_points); cv::Point2f origin; origin.x = img_cp.cols
  // - 110; origin.y = 20; cv::putText(img_cp, text, origin,
  // cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc) {
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] +
            w_ref_bl * img_ptr[width * 3] +
            w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] +
            w_ref_bl * img_ptr[1 + width * 3] +
            w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] +
            w_ref_bl * img_ptr[2 + width * 3] +
            w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap() {
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path =
      std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";

  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);

  Eigen::Quaterniond q(new_frame_->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap << cnt << " " << std::fixed
              << std::setprecision(6) // 保证浮点数精度为6位
              << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
              << t.x() << " " << t.y() << " " << t.z() << " " << 1
              << " " // CAMERA_ID (假设相机ID为1)
              << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

// void VIOManager::processFrame(
//     cv::Mat &img, multimap<double, pointWithVar> &pg,
//     const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
//     double img_time) {

//   // img_test = img.clone();

//   if (width != img.cols || height != img.rows) {
//     if (img.empty())
//       printf("[ VIO ] Empty Image!\n");
//     cv::resize(img, img,
//                cv::Size(img.cols * image_resize_factor,
//                         img.rows * image_resize_factor),
//                0, 0, CV_INTER_LINEAR);
//   }
//   img_rgb = img.clone();
//   img_cp = img.clone();

//   if (img.channels() == 3)
//     cv::cvtColor(img, img, CV_BGR2GRAY);

//   new_frame_.reset(new Frame(cam, img));
//   updateFrameState(*state);

//   resetGrid();

//   double t1 = omp_get_wtime();

//   retrieveFromVisualSparseMap(img, pg, feat_map);

//   double t2 = omp_get_wtime();

//   computeJacobianAndUpdateEKF(img);

//   double t3 = omp_get_wtime();

//   generateVisualMapPoints(img_rgb, pg);
//   // generateVisualMapPoints(img, pg);

//   double t4 = omp_get_wtime();

//   plotTrackedPoints();

//   if (plot_flag)
//     projectPatchFromRefToCur(feat_map);

//   double t5 = omp_get_wtime();

//   updateVisualMapPoints(img);

//   double t6 = omp_get_wtime();

//   updateReferencePatch(feat_map);

//   double t7 = omp_get_wtime();

//   if (colmap_output_en)
//     dumpDataForColmap();

//   frame_count++;
//   ave_total = ave_total * (frame_count - 1) / frame_count +
//               (t7 - t1 - (t5 - t4)) / frame_count;

//   // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
//   // printf("\033[1;32m[ VIO time ]: current frame:
//   retrieveFromVisualSparseMap
//   // time: %.6lf secs.\033[0m\n", t2 - t1); printf("\033[1;32m[ VIO time ]:
//   // current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H:
//   %.6lf
//   // secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
//   // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints
//   // time: %.6lf secs.\033[0m\n", t4 - t3); printf("\033[1;32m[ VIO time ]:
//   // current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 -
//   t5);
//   // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch
//   time:
//   // %.6lf secs.\033[0m\n", t7 - t6); printf("\033[1;32m[ VIO time ]:
//   current
//   // total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1
//   - (t5
//   // - t4), ave_total);

//   // ave_build_residual_time = ave_build_residual_time * (frame_count - 1)
//   /
//   // frame_count + (t2 - t1) / frame_count; ave_ekf_time = ave_ekf_time *
//   // (frame_count - 1) / frame_count + (t3 - t2) / frame_count;

//   // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time
//   <<
//   // RESET << endl; cout << BLUE << "ave_ekf_time: " << ave_ekf_time <<
//   RESET
//   <<
//   // endl;

//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");
//   printf("\033[1;34m|                         VIO Time "
//          "   |\033[0m\n");
//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");
//   printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size",
//          feat_map.size());
//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");
//   printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
//          "Time (secs)");
//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n",
//   "retrieveFromVisualSparseMap",
//          t2 - t1);
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n",
//   "computeJacobianAndUpdateEKF",
//          t3 - t2);
//   printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian",
//          compute_jacobian_time);
//   printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF",
//          update_ekf_time);
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n",
//   "generateVisualMapPoints",
//          t4 - t3);
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints",
//          t6 - t5);
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch",
//          t7 - t6);
//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time",
//          t7 - t1 - (t5 - t4));
//   printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time",
//          ave_total);
//   printf("\033[1;34m+----------------------------------------------------------"
//          "---+\033[0m\n");

//   // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + "
//   HZ";
//   // cv::Point2f origin;
//   // origin.x = 20;
//   // origin.y = 20;
//   // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6,
//   // cv::Scalar(255, 255, 255), 1, 8, 0);
//   // cv::imwrite("/home/chunran/Desktop/raycasting/" +
//   // std::to_string(new_frame_->id_) + ".png", img_cp);
// }

// sync_packages에서 계산된 시간 오프셋을 VIOManager로 전달하는 함수
void VIOManager::setCameraTimeOffsets(
    const std::map<int, double> &time_offsets) {
  m_img_time_offsets_from_last_update.assign(m_cameras.size(), 0.0);
  for (const auto &pair : time_offsets) {
    int cam_idx = pair.first;
    double offset = pair.second;
    m_img_time_offsets_from_last_update[cam_idx] = offset;
  }
}

// 원본 외향 매개변수를 백업하는 함수
void VIOManager::backupOriginalExtrinsics() {
  original_Rci_vec = this->m_R_c_i_vec;
  original_Pci_vec = this->m_P_c_i_vec;
  original_Rcl_vec = this->m_R_c_l_vec;
  original_Pcl_vec = this->m_P_c_l_vec;
  // original_Jdphi_dR = this->Jdphi_dR;
  // original_Jdp_dR = this->Jdp_dR;
}

// 백업된 원본 외향 매개변수로 복원하는 함수
void VIOManager::restoreOriginalExtrinsics() {
  this->m_R_c_i_vec = original_Rci_vec;
  this->m_P_c_i_vec = original_Pci_vec;
  this->m_R_c_l_vec = original_Rcl_vec;
  this->m_P_c_l_vec = original_Pcl_vec;
}

void VIOManager::compensateExtrinsicsByTimeOffset(
    const std::vector<Pose6D> &imu_poses, int cam_idx) {
  setCameraByIndex(cam_idx);

  double target_offset_time = m_img_time_offsets_from_last_update[cam_idx];

  const auto &pose_at_kf = imu_poses.back();
  std::cout << "imu_poses leng: " << imu_poses.size() << std::endl;
  // M3D R_at_kf;
  // R_at_kf << MAT_FROM_ARRAY(pose_at_kf.rot);
  // V3D P_at_kf;
  // P_at_kf << VEC_FROM_ARRAY(pose_at_kf.pos);

  if ((target_offset_time - pose_at_kf.offset_time) > -1e-6) {
    std::cout << "[ DEBUG ] cam" << cam_idx << " is not compensated"
              << std::endl;
    return;
  }

  double time_ratio = target_offset_time / pose_at_kf.offset_time;

  Eigen::Matrix4d T_update_k;

  if (!en_pose_linear_interpolate_backprop) {

    auto it_kp = imu_poses.end() - 1;
    for (; it_kp != imu_poses.begin(); it_kp--) {
      if (it_kp->offset_time < target_offset_time) {
        break;
      }
    }
    // std::cout << "[ DEBUG ] cam" << cam_idx
    //           << " IMUpose length: " << imu_poses.end() - 1 - it_kp <<
    //           std::endl;

    auto head = it_kp;
    auto tail = it_kp + 1;

    double dt = target_offset_time - head->offset_time;
    double head_tail_dt = tail->offset_time - head->offset_time;
    double s = dt / head_tail_dt;

    M3D R_at_head;
    R_at_head << MAT_FROM_ARRAY(head->rot);
    V3D P_at_head;
    P_at_head << VEC_FROM_ARRAY(head->pos);
    M3D R_at_tail;
    R_at_tail << MAT_FROM_ARRAY(tail->rot);
    V3D P_at_tail;
    P_at_tail << VEC_FROM_ARRAY(tail->pos);

    // M3D R_at_img_time = Eigen::Quaterniond(R_at_head)
    //                         .slerp(s, Eigen::Quaterniond(R_at_tail))
    //                         .toRotationMatrix();
    // V3D P_at_img_time = (1.0 - s) * P_at_head + s * P_at_tail;

    Eigen::Matrix4d T_head = Eigen::Matrix4d::Identity();
    T_head.block<3, 3>(0, 0) = R_at_head;
    T_head.block<3, 1>(0, 3) = P_at_head;

    Eigen::Matrix4d T_tail = Eigen::Matrix4d::Identity();
    T_tail.block<3, 3>(0, 0) = R_at_tail;
    T_tail.block<3, 1>(0, 3) = P_at_tail;

    // T_{tail -> head}
    Eigen::Matrix4d T_relative = T_head.inverse() * T_tail;
    // 4x4 twist
    Eigen::Matrix4d xi_hat = T_relative.log();
    Eigen::Matrix4d xi_hat_interpolated = s * xi_hat;
    Eigen::Matrix4d T_relative_interpolated = xi_hat_interpolated.exp();
    Eigen::Matrix4d T_prop_k = T_head * T_relative_interpolated;

    // double time_ratio = target_offset_time / pose_at_kf.offset_time;

    if (en_error_se3_backprop) {
      Eigen::Matrix4d T_prop_1 = Eigen::Matrix4d::Identity();
      T_prop_1.block<3, 3>(0, 0) = state_propagat->rot_end;
      T_prop_1.block<3, 1>(0, 3) = state_propagat->pos_end;

      Eigen::Matrix4d T_update_1 = Eigen::Matrix4d::Identity();
      T_update_1.block<3, 3>(0, 0) = state->rot_end;
      T_update_1.block<3, 1>(0, 3) = state->pos_end;

      Eigen::Matrix4d delta_T_corr = T_prop_1.inverse() * T_update_1;
      Eigen::Matrix4d xi_corr_total = delta_T_corr.log();
      Eigen::Matrix4d xi_corr_k = time_ratio * xi_corr_total;
      Eigen::Matrix4d delta_T_corr_k = xi_corr_k.exp();

      T_update_k = T_prop_k * delta_T_corr_k;
    } else { // just IMU backprop
      T_update_k = T_prop_k;
    }

  } else {
    Eigen::Matrix4d T_state_prev = Eigen::Matrix4d::Identity();
    T_state_prev.block<3, 3>(0, 0) = state_prev->rot_end;
    T_state_prev.block<3, 1>(0, 3) = state_prev->pos_end;

    Eigen::Matrix4d T_update_1 = Eigen::Matrix4d::Identity();
    T_update_1.block<3, 3>(0, 0) = state->rot_end;
    T_update_1.block<3, 1>(0, 3) = state->pos_end;

    // T_{tail -> head}
    Eigen::Matrix4d T_relative = T_state_prev.inverse() * T_update_1;
    // 4x4 twist
    Eigen::Matrix4d xi_hat = T_relative.log();
    Eigen::Matrix4d xi_hat_interpolated = time_ratio * xi_hat;
    Eigen::Matrix4d T_relative_interpolated = xi_hat_interpolated.exp();

    T_update_k = T_state_prev * T_relative_interpolated;
  }

  M3D R_w_img = T_update_k.block<3, 3>(0, 0);
  V3D P_w_img = T_update_k.block<3, 1>(0, 3);
  M3D R_w_kf = state->rot_end;
  V3D P_w_kf = state->pos_end;

  M3D R_kf_img = R_w_kf.transpose() * R_w_img;
  V3D P_kf_img = R_w_kf.transpose() * (P_w_img - P_w_kf);

  M3D R_img_kf = R_kf_img.transpose();
  V3D P_img_kf = -R_img_kf * P_kf_img;

  this->m_R_c_i_vec[cam_idx] = original_Rci_vec[cam_idx] * R_img_kf;
  this->m_P_c_i_vec[cam_idx] =
      original_Rci_vec[cam_idx] * P_img_kf + original_Pci_vec[cam_idx];
  this->m_R_c_l_vec[cam_idx] = original_Rci_vec[cam_idx] * Rli.transpose();
  this->m_P_c_l_vec[cam_idx] =
      original_Pci_vec[cam_idx] +
      original_Rci_vec[cam_idx] * (-Rli.transpose() * Pli);

  setCameraByIndex(cam_idx);
}