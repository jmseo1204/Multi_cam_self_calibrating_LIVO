/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <Eigen/Eigenvalues>
#include <iterator>

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0), extR(M3D::Identity()) {
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_body_prev.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());

  // VIO 파라미터 읽기
  vio_manager->readParameters(nh);

  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh) {

  nh.param<bool>("common/en_sliding_window_ICP", en_sliding_window_ICP, false);
  nh.param<bool>("common/only_side_cam", only_side_cam, false);
  nh.param<bool>("common/mapping_after_vio", mapping_after_vio, false);
  nh.param<bool>("common/en_cam_backprop", en_cam_backprop, false);
  nh.param<double>("common/lidar_window_size", lidar_window_size, 0.1);
  nh.param<int>("common/minimum_simultaneous_frame_num",
                minimum_simultaneous_frame_num, 1);
  nh.param<int>("common/num_of_cam", num_of_cam, 1);
  if (num_of_cam > 1) {
    nh.getParam("common/img_topics", img_topics);
  } else {
    std::string single_topic;
    nh.param<std::string>("common/img_topic", single_topic, "");
    if (!single_topic.empty())
      img_topics.push_back(single_topic);
  }

  nh.getParam("extrin_calib/extrinsic_T", extrinT_il);
  nh.getParam("extrin_calib/extrinsic_R", extrinR_il);

  // nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT,
  //   vector<double>());
  // nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR,
  //     vector<double>());
  // nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT,
  // vector<double>()); nh.param<vector<double>>("extrin_calib/Rcl",
  // cameraextrinR, vector<double>());

  m_R_c_l_vec.resize(num_of_cam);
  m_P_c_l_vec.resize(num_of_cam);
  for (int i = 0; i < num_of_cam; ++i) {
    std::vector<double> R_cl_vec, P_cl_vec;
    nh.getParam("extrin_calib/R_cl_" + std::to_string(i), R_cl_vec);
    nh.getParam("extrin_calib/P_cl_" + std::to_string(i), P_cl_vec);
    m_R_c_l_vec[i] << MAT_FROM_ARRAY(R_cl_vec);
    m_P_c_l_vec[i] << VEC_FROM_ARRAY(P_cl_vec);
  }

  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");
  nh.param<double>("common/sync_time_window", sync_time_window, 0.075);

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled,
                 false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() {
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT_il);
  extR << MAT_FROM_ARRAY(extrinR_il);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT_il);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR_il);

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  // vio_manager->setImuToLidarExtrinsic(extT, extR);

  last_timestamp_imgs.resize(num_of_cam);
  vio_manager->m_cameras.resize(num_of_cam);

  std::cout << "m_cameras size: " << vio_manager->m_cameras.size() << std::endl;

  for (int i = 0; i < num_of_cam; ++i) {
    last_timestamp_imgs[i] = -1.0;
    if (!vk::camera_loader::loadFromRosNs("laserMapping",
                                          vio_manager->m_cameras[i]) &&
        !vk::camera_loader::loadFromRosNs("laserMapping/cam_" +
                                              std::to_string(i + 1),
                                          vio_manager->m_cameras[i]))
      throw std::runtime_error("Camera model not correctly specified.");
  }

  vio_manager->setExtrinsicParameters(extR, extT, m_R_c_l_vec, m_P_c_l_vec);

  vio_manager->sub_feat_maps = vector<unordered_map<VOXEL_LOCATION, int> *>(
      num_of_cam, new unordered_map<VOXEL_LOCATION, int>());

  vio_manager->state = &_state;
  vio_manager->state_prev = &_state_prev;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  std::cout << "Initialize VIO" << std::endl;
  vio_manager->initializeVIO();
  std::cout << "Initialize VIO end" << std::endl;

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en)
    p_imu->disable_imu();
  if (!gravity_est_en)
    p_imu->disable_gravity_est();
  if (!ba_bg_est_en)
    p_imu->disable_bias_est();
  if (!exposure_estimate_en)
    p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() {
  if (pcd_save_en && colmap_output_en) {
    const std::string folderPath =
        std::string(ROOT_DIR) + "/scripts/colmap_output.sh";

    std::string chmodCommand = "chmod +x " + folderPath;

    int chmodRet = system(chmodCommand.c_str());
    if (chmodRet != 0) {
      std::cerr << "Failed to set execute permissions for the script."
                << std::endl;
      return;
    }

    int executionRet = system(folderPath.c_str());
    if (executionRet != 0) {
      std::cerr << "Failed to execute the script." << std::endl;
      return;
    }
  }
  if (colmap_output_en)
    fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt",
                     std::ios::out);
  if (pcd_save_interval > 0)
    fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json",
                      std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(
    ros::NodeHandle &nh, image_transport::ImageTransport &it) {
  sub_pcl =
      p_pre->lidar_type == AVIA
          ? nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this)
          : nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  // sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);

  m_img_buffers.resize(num_of_cam);
  m_img_time_buffers.resize(num_of_cam);
  sub_imgs.resize(num_of_cam);
  for (int i = 0; i < num_of_cam; ++i) {
    sub_imgs[i] = nh.subscribe<sensor_msgs::Image>(
        img_topics[i], 200000, boost::bind(&LIVMapper::img_cbk, this, _1, i));
    ROS_INFO("Subscribing to image topic [%d]: %s", i, img_topics[i].c_str());
  }

  pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>(
      "visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>(
      "/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed =
      nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg =
      nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher =
      nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImuPropOdom =
      nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer =
      nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_ =
      nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
  pubImages.resize(num_of_cam);
  for (int i = 0; i < num_of_cam; ++i) {
    pubImages[i] = it.advertise("/rgb_img" + std::to_string(i), 1);
  }
}
void LIVMapper::handleFirstFrame() {
  if (!is_first_frame) {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment(StatesGroup &state) {
  if (!p_imu->imu_need_init && !gravity_align_finished) {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    state.pos_end = G_R_I0 * state.pos_end;
    state.rot_end = G_R_I0 * state.rot_end;
    state.vel_end = G_R_I0 * state.vel_end;
    state.gravity = G_R_I0 * state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

bool LIVMapper::processImu() {
  // double t0 = omp_get_wtime();

  _state_prev = _state;
  p_imu->Process2(LidarMeasures, _state, last_IMU_state, feats_undistort);
  if (p_imu->imu_need_init)
    return false;

  if (gravity_align_en) {
    gravityAlignment(_state); // 중력방향으로 z축 맞추는거 할지말지
    // gravityAlignment(last_IMU_state);
  }

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  publish_odometry(pubImuPredictedOdom, last_IMU_state);

  return true;

  // double t_prop = omp_get_wtime();
  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() <<
  // std::endl; std::cout << "[ Mapping ] predict cov: " <<
  // _state.cov.diagonal().transpose() << std::endl; std::cout << "[ Mapping ]
  // predict sta: " << state_propagat.pos_end.transpose() <<
  // state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() {
  switch (LidarMeasures.lio_vio_flg) {
  case VIO:
    handleVIO();
    break;
  case LIO:
  case LO:
    handleLIO();
    break;
  }
}

void LIVMapper::handleVIO() {
  std::cout << "handleVIO" << std::endl;

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20)
           << LidarMeasures.last_lio_update_time - _first_lidar_time << " "
           << euler_cur.transpose() * 57.3 << " " << _state.pos_end.transpose()
           << " " << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " " << _state.bias_a.transpose()
           << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;

  auto &m = LidarMeasures.measures.back();

  if (m.imgs.empty()) {
    return;
  }

  // if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) {
  //   return;
  // }

  const auto &imu_poses = p_imu->IMUpose_cp;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) -
           plot_time) < (frame_cnt / 2 * 0.1)) {
    vio_manager->plot_flag = true;
  } else {
    vio_manager->plot_flag = false;
  }

  // Snapshot covariances before VIO
  Eigen::Matrix3d rot_cov_pre = _state.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d trans_cov_pre = _state.cov.block<3, 3>(3, 3);

  vio_manager->processMultiCamVIO(
      m.imgs, m.img_camera_indices, imu_poses, _pv_prev,
      voxelmap_manager->voxel_map_,
      LidarMeasures.last_lio_update_time - _first_lidar_time, en_cam_backprop);

  // Write VIO stats
  Eigen::Matrix3d rot_cov_post = _state.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d trans_cov_post = _state.cov.block<3, 3>(3, 3);
  writeVIOStats(LidarMeasures.last_lio_update_time, rot_cov_pre, trans_cov_pre,
                rot_cov_post, trans_cov_post, num_of_cam, m.img_camera_indices);

  // if (mapping_after_vio) {
  //   PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  //   transformLidar(_state.rot_end, _state.pos_end, feats_down_body,
  //                  world_lidar);
  //   for (size_t i = 0; i < world_lidar->points.size(); i++) {
  //     voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x,
  //         world_lidar->points[i].y, world_lidar->points[i].z;
  //     M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
  //     M3D var = voxelmap_manager->body_cov_list_[i];
  //     var =
  //         (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose()
  //         +
  //         (-point_crossmat) * _state.cov.block<3, 3>(0, 0) *
  //             (-point_crossmat).transpose() +
  //         _state.cov.block<3, 3>(3, 3);
  //     voxelmap_manager->pv_list_[i].var = var;
  //   }

  //   voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  //   if (voxelmap_manager->config_setting_.map_sliding_en) {
  //     voxelmap_manager->mapSliding();
  //   }

  //   std::cout << "[ AFTER VIO ] Update Voxel Map" << std::endl;

  //   _pv_list = voxelmap_manager->pv_list_;

  //   for (const auto &point : _pv_list) {
  //     auto it = _pv_prev.find(point.timestamp);
  //     if (it == _pv_prev.end()) {
  //       continue;
  //     } else {
  //       auto &pv = it->second;
  //       pv.point_w << point.point_w(0), point.point_w(1), point.point_w(2);
  //       pv.var = point.var;
  //     }
  //   }

  //   PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI());

  //   if (dense_map_en) {
  //     PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
  //     int size = laserCloudFullRes->points.size();
  //     laserCloudWorld = PointCloudXYZI::Ptr(new PointCloudXYZI(size, 1));
  //     for (int i = 0; i < size; i++) {
  //       RGBpointBodyToWorld(&laserCloudFullRes->points[i],
  //                           &laserCloudWorld->points[i]);
  //     }

  //   } else {
  //     for (const auto &pair : _pv_prev) {
  //       const auto &pv = pair.second;
  //       PointType point;
  //       point.x = pv.point_w(0);
  //       point.y = pv.point_w(1);
  //       point.z = pv.point_w(2);
  //       // point.intensity = pv.intensity;
  //       // point.curvature = pv.timestamp;
  //       laserCloudWorld->points.push_back(point);
  //     }
  //   }

  //   // std::cout << "handleLIO _pv_prev size: " << _pv_prev.size() <<
  //   std::endl;

  //   *pcl_w_wait_pub = *laserCloudWorld;
  // }

  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size()
            << std::endl;
  *pcl_wait_pub += *pcl_w_wait_pub;
  bool publish_frame = false;
  if (pub_num == pub_scan_num) {
    pub_num = 1;
    publish_frame = true;
  } else {
    pub_num++;
  }

  for (size_t i = 0; i < m.img_camera_indices.size(); ++i) {
    int cam_idx = m.img_camera_indices[i];
    vio_manager->setCameraByIndex(cam_idx);
    publish_img_rgb(pubImages[cam_idx], vio_manager, cam_idx);
    publish_frame_world(publish_frame, pubLaserCloudFullRes, vio_manager,
                        cam_idx);
  }
  if (publish_frame) {
    PointCloudXYZI().swap(*pcl_wait_pub);
  }
  PointCloudXYZI().swap(*pcl_w_wait_pub);

  if (imu_prop_enable) { // EKF 상태는 LIO 결과로 갱신
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20)
           << LidarMeasures.last_lio_update_time - _first_lidar_time << " "
           << euler_cur.transpose() * 57.3 << " " << _state.pos_end.transpose()
           << " " << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " " << _state.bias_a.transpose()
           << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
           << feats_undistort->points.size() << std::endl;
}

void LIVMapper::writeVIOStats(double cur_time,
                              const Eigen::Matrix3d &rot_cov_pre,
                              const Eigen::Matrix3d &trans_cov_pre,
                              const Eigen::Matrix3d &rot_cov_post,
                              const Eigen::Matrix3d &trans_cov_post,
                              int total_cam_count,
                              const std::vector<int> &used_cam_indices) {
  // Prepare VIO stats file (open once and write header)
  if (!fout_vio_stats.is_open()) {
    std::string path =
        vio_stats_path.empty()
            ? std::string("/mnt/c/Users/USER/Desktop/fast_livo2/results/") +
                  "vio_stats.txt"
            : vio_stats_path;
    fout_vio_stats.open(path, std::ios::out | std::ios::app);
  }

  // Header: write once (columns description)
  if (!vio_stats_header_written) {
    fout_vio_stats << "# time(s), pre_rot_eig1, pre_rot_eig2, pre_rot_eig3, "
                   << "pre_trans_eig1, pre_trans_eig2, pre_trans_eig3, "
                   << "post_rot_eig1, post_rot_eig2, post_rot_eig3, "
                   << "post_trans_eig1, post_trans_eig2, post_trans_eig3";
    // Camera usage columns cam_used[0..N-1]
    for (int cam = 0; cam < total_cam_count; ++cam) {
      fout_vio_stats << ", cam_used[" << cam << "]";
    }
    // Only level 0 average total rows
    fout_vio_stats << ", level0_avg_total_rows";
    fout_vio_stats << std::endl;
    vio_stats_header_written = true;
  }

  // Compute eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot_pre(rot_cov_pre);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans_pre(trans_cov_pre);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot_post(rot_cov_post);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans_post(trans_cov_post);

  Eigen::Vector3d rot_eigs_pre = es_rot_pre.eigenvalues();
  Eigen::Vector3d trans_eigs_pre = es_trans_pre.eigenvalues();
  Eigen::Vector3d rot_eigs_post = es_rot_post.eigenvalues();
  Eigen::Vector3d trans_eigs_post = es_trans_post.eigenvalues();

  // Data row
  fout_vio_stats << std::setprecision(9) << std::fixed << cur_time << ", "
                 << rot_eigs_pre(0) << ", " << rot_eigs_pre(1) << ", "
                 << rot_eigs_pre(2) << ", " << trans_eigs_pre(0) << ", "
                 << trans_eigs_pre(1) << ", " << trans_eigs_pre(2) << ", "
                 << rot_eigs_post(0) << ", " << rot_eigs_post(1) << ", "
                 << rot_eigs_post(2) << ", " << trans_eigs_post(0) << ", "
                 << trans_eigs_post(1) << ", " << trans_eigs_post(2);

  // Camera usage flags
  std::vector<int> cam_flags(static_cast<size_t>(total_cam_count), 0);
  for (int idx : used_cam_indices) {
    if (idx >= 0 && idx < total_cam_count)
      cam_flags[static_cast<size_t>(idx)] = 1;
  }
  for (int cam = 0; cam < total_cam_count; ++cam) {
    fout_vio_stats << ", " << cam_flags[static_cast<size_t>(cam)];
  }

  // Append only level 0 average (convert rows to point count by dividing by
  // patch_size_total)
  double avg_level0_points = 0.0;
  if (!vio_manager->level_avg_visual_points.empty()) {
    avg_level0_points = vio_manager->level_avg_visual_points[0];
  }
  fout_vio_stats << ", " << avg_level0_points;
  fout_vio_stats << std::endl;
}

void LIVMapper::handleLIO() {
  std::cout << "handleLIO" << std::endl;

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time
           << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose()
           << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " "
           << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;

  if (feats_undistort->empty() || (feats_undistort == nullptr)) {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);

  double t_down = omp_get_wtime();

  if (!lidar_map_inited) {
    lidar_map_inited = true;
    voxelmap_manager->feats_down_body_ = feats_down_body;
    transformLidar(_state.rot_end, _state.pos_end, feats_down_body,
                   feats_down_world);
    voxelmap_manager->feats_down_world_ = feats_down_world;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  int idx_offset = 0;

  MeasureGroup &meas = LidarMeasures.measures.back();
  while (!(_pv_prev.empty() ||
           (_pv_prev.begin()->first + lidar_window_size) >= meas.lio_time)) {
    _pv_prev.erase(_pv_prev.begin());
  }

  if (en_sliding_window_ICP) {

    // for (int i = 0; i < feats_down_body_prev->points.size(); i++) {
    //   feats_down_body_prev->points[i].curvature -=
    //       LidarMeasures.last_lio_update_time - prev_lio_update_time;
    // }
    // while (!(feats_down_body_prev->points.empty() ||
    //          (feats_down_body_prev->points.begin().curvature / 1000.0 +
    //           lidar_window_size) >=
    //              meas.lio_time - LidarMeasures.last_lio_update_time)) {
    //   feats_down_body_prev->points.erase(feats_down_body_prev->points.begin());
    // }
    // idx_offset = feats_down_body_prev->points.size();
    // *feats_down_body_prev = *feats_down_body_prev + *feats_down_body;

    idx_offset = _pv_prev.size();

    PointCloudXYZI::Ptr feats_down_world_prev(new PointCloudXYZI());
    for (const auto &pair : _pv_prev) {
      const pointWithVar &pv = pair.second;
      PointType point;
      point.x = pv.point_w(0);
      point.y = pv.point_w(1);
      point.z = pv.point_w(2);
      point.curvature = (pair.first - LidarMeasures.last_lio_update_time) *
                        1000.0; // it can be negative
      // point.intensity = 0.0;
      feats_down_world_prev->points.push_back(point);
    }

    feats_down_body_prev.reset(new PointCloudXYZI());
    M3D R_lw((_state.rot_end * extR).transpose());
    V3D T_li(-extR.transpose() * extT);

    transformLidar(R_lw, -R_lw * (_state.rot_end * extT + _state.pos_end),
                   feats_down_world_prev, feats_down_body_prev, false);

    *feats_down_body_prev = *feats_down_body_prev + *feats_down_body;

  } else {
    *feats_down_body_prev = *feats_down_body;
  }

  // std::cout << "handleLIO1" << std::endl;

  feats_down_size = feats_down_body_prev->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body_prev;
  voxelmap_manager->feats_down_size_ = feats_down_size;

  voxelmap_manager->StateEstimation(state_propagat, LidarMeasures);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  // std::cout << "handleLIO2" << std::endl;

  prev_lio_update_time = LidarMeasures.last_lio_update_time;
  LidarMeasures.last_lio_update_time = meas.lio_time;
  std::cout << std::setprecision(7) << std::fixed
            << "last_lio_update_time updated: "
            << LidarMeasures.last_lio_update_time << std::endl;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt",
                   std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open())
        ROS_ERROR("open fail\n");
    } else {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt",
                   std::ios::app);
      if (!evoFile.is_open())
        ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0]
            << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << std::endl;
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1),
                                                    euler_cur(2));
  publish_odometry(pubOdomAftMapped, _state);

  double t3 = omp_get_wtime();

  // PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  // transformLidar(_state.rot_end * extR, _state.rot_end * extT +
  // _state.pos_end, feats_down_body, world_lidar);

  // std::vector<pointWithVar> temp_pv_list(voxelmap_manager->pv_list_.begin() +
  //                                            idx_offset,
  //                                        voxelmap_manager->pv_list_.end());
  // voxelmap_manager->pv_list_.swap(temp_pv_list);
  // std::vector<pointWithVar>().swap(temp_pv_list);

  // std::vector<M3D> temp_cross(voxelmap_manager->cross_mat_list_.begin() +
  //                                 idx_offset,
  //                             voxelmap_manager->cross_mat_list_.end());
  // voxelmap_manager->cross_mat_list_.swap(temp_cross);
  // std::vector<M3D>().swap(temp_cross);

  // std::vector<M3D> temp_body(voxelmap_manager->body_cov_list_.begin() +
  //                                idx_offset,
  //                            voxelmap_manager->body_cov_list_.end());
  // voxelmap_manager->body_cov_list_.swap(temp_body);
  // std::vector<M3D>().swap(temp_body);

  // for (size_t i = 0; i < world_lidar->points.size(); i++) {
  //   voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x,
  //       world_lidar->points[i].y, world_lidar->points[i].z;
  //   voxelmap_manager->pv_list_[i].timestamp =
  //       prev_lio_update_time + world_lidar->points[i].curvature / 1000.0;
  //   M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
  //   M3D var = voxelmap_manager->body_cov_list_[i];
  //   var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose()
  //   +
  //         (-point_crossmat) * _state.cov.block<3, 3>(0, 0) *
  //             (-point_crossmat).transpose() +
  //         _state.cov.block<3, 3>(3, 3);
  //   voxelmap_manager->pv_list_[i].var = var;

  //   _pv_prev.insert({_pv_list[i].timestamp, _pv_list[i]});
  // }

  for (auto it = _pv_list.begin() + idx_offset; it != _pv_list.end(); ++it) {
    auto &pv = *it;
    PointType point;
    point.x = pv.point_w(0);
    point.y = pv.point_w(1);
    point.z = pv.point_w(2);
    _pv_prev.insert({pv.timestamp, pv});
  }

  // std::cout << "handleLIO3" << std::endl;

  if (slam_mode_ == ONLY_LIO || !mapping_after_vio) {
    voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
    std::cout << "[ LIO ] Update Voxel Map" << std::endl;

    if (voxelmap_manager->config_setting_.map_sliding_en) {
      voxelmap_manager->mapSliding();
    }
  }

  _pv_list = voxelmap_manager->pv_list_;

  // if (en_sliding_window_ICP)
  //   _pv_prev.clear();

  // std::cout << "handleLIO4" << std::endl;

  assert(_pv_prev.begin()->first <= (_pv_prev.begin() + idx_offset)->first);
  auto it = _pv_prev.begin();
  // for (size_t i = 0; i < _pv_prev.size(); i++) {
  //   // assert(_pv_prev[i - 1].first <= _pv_prev[i].first);
  //   it++;
  //   if (i % 1000 == 0)
  //     std::cout << it->first << std::endl;
  // }

  // while (!(_pv_prev.empty() || _pv_prev.rbegin()->first < 3000000000.0)) {
  //   auto it = _pv_prev.end();
  //   --it;
  //   _pv_prev.erase(it);
  // }

  double t4 = omp_get_wtime();

  if (slam_mode_ == ONLY_LIO || !mapping_after_vio) {

    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI());

    if (dense_map_en) {
      PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
      int size = laserCloudFullRes->points.size();
      laserCloudWorld = PointCloudXYZI::Ptr(new PointCloudXYZI(size, 1));
      for (int i = 0; i < size; i++) {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                            &laserCloudWorld->points[i]);
      }

    } else {
      for (const auto &pair : _pv_prev) {
        const auto &pv = pair.second;
        PointType point;
        point.x = pv.point_w(0);
        point.y = pv.point_w(1);
        point.z = pv.point_w(2);
        // point.intensity = pv.intensity;
        // point.curvature = pv.timestamp;
        laserCloudWorld->points.push_back(point);
      }
    }

    // std::cout << "handleLIO _pv_prev size: " << _pv_prev.size() << std::endl;

    *pcl_w_wait_pub = *laserCloudWorld;
  }

  if (!img_en)
    // std::cout << "handleLIO publish_frame_world: "
    //           << pcl_w_wait_pub->points.size() << std::endl;
    publish_frame_world(true, pubLaserCloudFullRes, vio_manager, -1);
  if (pub_effect_point_en)
    publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_)
    voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu =
      aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) /
  // frame_num; aver_time_map_inre = aver_time_map_inre * (frame_num - 1) /
  // frame_num + (t4 - t3) / frame_num; aver_time_solve = aver_time_solve *
  // (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) /
  // frame_num + solve_const_H_time / frame_num; printf("[ mapping time ]: per
  // scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP:
  // %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total:
  //         %0.6f \n", t_prop - t0, t1 - t_prop, match_time, solve_time, t3 -
  //         t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time,
  //         aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs,
  // map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map
  //         incre: %0.6f secs, total: %0.6f secs.\033[0m\n", t2 - t1, t4 -
  //         t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                 "
         "   |\033[0m\n");
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time",
         aver_time_consu);
  printf("\033[1;34m+----------------------------------------------------------"
         "---+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20)
           << LidarMeasures.last_lio_update_time - _first_lidar_time << " "
           << euler_cur.transpose() * 57.3 << " " << _state.pos_end.transpose()
           << " " << _state.vel_end.transpose() << " "
           << _state.bias_g.transpose() << " " << _state.bias_a.transpose()
           << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " "
           << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() {
  if (pcd_save_en &&
      (pcl_wait_save->points.size() > 0 ||
       pcl_wait_save_intensity->points.size() > 0) &&
      pcd_save_interval < 0) {
    std::string raw_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd,
                               filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);

      pcd_writer.writeBinary(raw_points_dir,
                             *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
                << " with point count: " << pcl_wait_save->points.size()
                << RESET << std::endl;

      pcd_writer.writeBinary(
          downsampled_points_dir,
          *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: "
                << downsampled_points_dir
                << " with point count after filtering: "
                << downsampled_cloud->points.size() << RESET << std::endl;

      if (colmap_output_en) {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) {
          const auto &point = downsampled_cloud->points[i];
          fout_points << i << " " << std::fixed << std::setprecision(6)
                      << point.x << " " << point.y << " " << point.z << " "
                      << static_cast<int>(point.r) << " "
                      << static_cast<int>(point.g) << " "
                      << static_cast<int>(point.b) << " " << 0 << std::endl;
        }
      }
    } else {
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
                << " with point count: "
                << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() {
  // 초기에 TUM trajectory 파일 경로 생성
  // std::string tum_output_path = std::string(ROOT_DIR) + "/Log/" +
  //                               std::to_string(ros::Time::now().toSec()) +
  //                               "_tum.txt";
  std::string tum_output_path =
      "/mnt/c/Users/USER/Desktop/sliding_fast_livo2/results/" +
      std::to_string(int(ros::Time::now().toSec()) % 10000) + ".txt";

  std::string stat_output_path =
      "/mnt/c/Users/USER/Desktop/sliding_fast_livo2/results/" +
      std::to_string(int(ros::Time::now().toSec()) % 10000) + "_stat.txt";

  // set the VIO stats output path for writeVIOStats
  vio_stats_path = stat_output_path;

  ros::Rate rate(5000);
  while (ros::ok()) {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    if (!processImu()) {
      LidarMeasures.lio_vio_flg = VIO;
      continue;
    }

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
    if (!img_en || LidarMeasures.lio_vio_flg == VIO)
      save_path_to_file(tum_output_path);
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt,
                              V3D acc_avr, V3D angvel_avr) {
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr +
                V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1],
                    imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end +
                           imu_prop_state.vel_end * dt +
                           0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e) {
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) {
    return;
  }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty()) {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg) {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() &&
              prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time)) {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++) {
        double t_from_lidar_end_time =
            prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " <<
        // last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x,
                    prop_imu_buffer[i].linear_acceleration.y,
                    prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x,
                    prop_imu_buffer[i].angular_velocity.y,
                    prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    } else {
      V3D acc_imu(newest_imu.linear_acceleration.x,
                  newest_imu.linear_acceleration.y,
                  newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y,
                  newest_imu.angular_velocity.z);
      double t_from_lidar_end_time =
          newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot,
                               const Eigen::Vector3d t,
                               const PointCloudXYZI::Ptr &input_cloud,
                               PointCloudXYZI::Ptr &trans_cloud,
                               bool default_il) {
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (default_il ? (extR * p + extT) : p) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    pi.curvature = p_c.curvature;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po) {
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T>
void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi,
                                 Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T>
Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi) {
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi,
                                    PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
  po->curvature = pi->curvature;
}

void LIVMapper::standard_pcl_cbk(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  if (!lidar_en)
    return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(
    const livox_ros_driver::CustomMsg::ConstPtr &msg_in) {
  if (!lidar_en)
    return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(
      new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 &&
  // last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() -
  //   last_timestamp_lidar); sync_jump_flag = true; msg->header.stamp =
  //   ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 &&
      !imu_buffer.empty()) {
    double timediff_imu_wrt_lidar =
        last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m",
           timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  if (!imu_en)
    return;

  if (last_timestamp_lidar < 0.0)
    return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp =
      ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en)) {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n",
             last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en)
    timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu) {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp -
  //   last_timestamp_imu); mtx_buffer.unlock(); sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable) {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) {
      prop_imu_buffer.push_back(*msg);
    }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in, int cam_idx) {
  if (!img_en)
    return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // std::cout << "[DEBUG] img_cbk called! timestamp: " <<
  // msg->header.stamp.toSec() << std::endl; if
  // ((abs(msg->header.stamp.toSec()
  // - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() -
  //   last_timestamp_img); sync_jump_flag = true; msg->header.stamp =
  //   ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en) {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0)
      return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() - img_time_offset;
  if (abs(msg_header_time - last_timestamp_imgs[cam_idx]) < 0.001)
    return;
  ROS_INFO("[CAM %d] Get image, its header time: %.6f", cam_idx,
           msg_header_time);
  if (last_timestamp_lidar < 0)
    return;

  if (msg_header_time < last_timestamp_imgs[cam_idx]) {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_imgs[cam_idx] < 0.02) {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  m_img_buffers[cam_idx].push_back(img_cur);
  m_img_time_buffers[cam_idx].push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_imgs[cam_idx] = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas) {
  if (lid_raw_data_buffer.empty() && lidar_en)
    return false;
  if (imu_buffer.empty() && imu_en)
    return false;

  switch (slam_mode_) {
  case ONLY_LIO: {
    if (meas.last_lio_update_time < 0.0)
      meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed) {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar
      if (meas.lidar->points.size() <= 1)
        return false;

      meas.lidar_frame_beg_time =
          lid_header_time_buffer.front(); // generate lidar_frame_beg_time
      meas.lidar_frame_end_time =
          meas.lidar_frame_beg_time +
          meas.lidar->points.back().curvature / double(1000); // calc lidar
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true; // flag
    }

    if (imu_en &&
        last_timestamp_imu <
            meas.lidar_frame_end_time) { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty()) {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time)
        break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg =
        LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO: {
    double max_time_start = std::numeric_limits<double>::min();
    double min_time_start = std::numeric_limits<double>::max();
    for (int i = 0; i < m_img_buffers.size(); i++) {
      // DEBUG
      if (only_side_cam && (i == 0 || i == 2)) {
        continue;
      }

      while (!m_img_time_buffers[i].empty() &&
             m_img_time_buffers[i].front() <
                 meas.last_lio_update_time + 0.00001) {
        m_img_time_buffers[i].pop_front();
        m_img_buffers[i].pop_front();
      }

      if (m_img_time_buffers[i].empty()) {
        // continue;
        return false;
      } else {

        max_time_start = max_time_start < m_img_time_buffers[i].back()
                             ? m_img_time_buffers[i].back()
                             : max_time_start;
        min_time_start = min_time_start > m_img_time_buffers[i].front()
                             ? m_img_time_buffers[i].front()
                             : min_time_start;
        // min_time_start = m_img_time_buffers[0].front();
      }
    }
    if (max_time_start - min_time_start < sync_time_window)
      return false;

    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    switch (last_lio_vio_flg) {
    case WAIT:
    case VIO: {

      if (meas.last_lio_update_time <= 0.0)
        meas.last_lio_update_time = lid_header_time_buffer.front() - 0.00001;

      // double end_time = min_time_start + sync_time_window;

      std::map<int, std::pair<double, cv::Mat>> candidates;

      double key_frame_time;
      vector<double> key_frame_times;

      // [DEBUG]
      // const char block = '#';
      // const double time_unit = 0.01;

      // vector<double> fronts;
      // for (const auto &buf : m_img_time_buffers) {
      //   if (!buf.empty()) {
      //     fronts.push_back(buf.front());
      //   } else {
      //     fronts.push_back(numeric_limits<double>::max());
      //   }
      // }
      // double min_time = *min_element(fronts.begin(), fronts.end());

      // for (size_t i = 0; i < m_img_time_buffers.size(); ++i) {
      //   size_t size = m_img_time_buffers[i].size();
      //   if (size == 0) {
      //     cout << "Buffer " << i << ": (empty)" << endl;
      //     continue;
      //   }

      //   double offset = fronts[i] - min_time;
      //   int spaces = static_cast<int>(round(offset / time_unit));

      //   cout << "Buffer " << i << ": ";
      //   cout << string(spaces, ' ');
      //   cout << string(size, block);
      //   cout << " (" << size << ", offset: " << fixed << setprecision(6)
      //        << offset << "s)" << endl;
      // }

      double end_time = min_time_start + sync_time_window;

      int min_idx = -1;
      for (int i = 0; i < num_of_cam; i++) {

        // DEBUG
        if (only_side_cam && (i == 0 || i == 2)) {
          continue;
        }

        if (m_img_time_buffers[i].empty()) {
          std::cout << "channel pass(empty): " << i << std::endl;
          continue;
        }

        double current_img_time = m_img_time_buffers[i].front();

        if (current_img_time >= end_time) {
          std::cout << std::fixed << std::setprecision(4)
                    << "channel pass(over): " << i << " - " << current_img_time
                    << std::endl;
          continue;
        }

        double capture_time = current_img_time + exposure_time_init;
        cv::Mat image = m_img_buffers[i].front();
        candidates[i] = {capture_time, image};
        key_frame_times.push_back(capture_time);
        min_idx = min_idx == -1
                      ? i
                      : (m_img_time_buffers[min_idx].front() > current_img_time
                             ? i
                             : min_idx);
      }

      if (candidates.empty()) {
        std::cout << "No Candidates" << std::endl;
        return false;
      } else if (candidates.size() < minimum_simultaneous_frame_num) {
        std::cout << "Below minimum Candidates" << std::endl;
        m_img_time_buffers[min_idx].pop_front();
        m_img_buffers[min_idx].pop_front();
        return false;
      }

      key_frame_time =
          *std::max_element(key_frame_times.begin(), key_frame_times.end());

      m_candidates = candidates;

      std::cout << "candidates: " << candidates.size() << std::endl;

      std::map<int, double> time_offsets_from_last_update;
      for (const auto &cand : m_candidates) {
        int cam_idx = cand.first;
        double capture_time = cand.second.first;
        time_offsets_from_last_update[cam_idx] =
            capture_time - meas.last_lio_update_time;
      }
      vio_manager->setCameraTimeOffsets(time_offsets_from_last_update);

      meas.measures.clear();

      double lid_newest_time =
          lid_header_time_buffer.back() +
          lid_raw_data_buffer.back()->points.back().curvature / 1000.0;
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      if (key_frame_time > lid_newest_time) {
        std::cout << std::fixed << std::setprecision(6)
                  << "[sync false] lid < key_frame: " << lid_newest_time << "<"
                  << key_frame_time << std::endl;
        return false;
      }

      if (key_frame_time > imu_newest_time) {
        std::cout << std::fixed << std::setprecision(6)
                  << "[sync false] IMU < key_frame: " << imu_newest_time << "<"
                  << key_frame_time << std::endl;
        return false;
      }

      for (int i = 0; i < num_of_cam; i++) {
        if (candidates.count(i)) {
          m_img_time_buffers[i].pop_front();
          m_img_buffers[i].pop_front();
        }
      }

      if ((meas.pcl_proc_next->size() == 0 &&
           key_frame_time < lid_header_time_buffer.front())) {
        std::cout << "No pcl pending" << std::endl;
        return false;
      }

      struct MeasureGroup m;

      m.imu.clear();
      m.lio_time = key_frame_time;
      m.vio_time = key_frame_time;
      mtx_buffer.lock();

      while (!imu_buffer.empty() && imu_buffer.front()->header.stamp.toSec() <
                                        meas.last_lio_update_time) {
        imu_buffer.pop_front(); // 오래된 IMU 데이터 제거
      }
      while (!imu_buffer.empty() &&
             imu_buffer.front()->header.stamp.toSec() <= key_frame_time) {
        m.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);

      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);

      while (!lid_raw_data_buffer.empty()) {
        if (lid_header_time_buffer.front() > key_frame_time)
          break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());

        for (int i = 0; i < pcl.size(); i++) {
          auto pt = pcl[i];
          pt.curvature +=
              (frame_header_time - meas.last_lio_update_time) * 1000.0f;
          meas.pcl_proc_cur->points.push_back(pt);
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      while (!meas.pcl_proc_cur->points.empty() &&
             meas.pcl_proc_cur->points.back().curvature >
                 (m.lio_time - meas.last_lio_update_time) * 1000.0f) {
        auto pt = meas.pcl_proc_cur->points.back();
        pt.curvature += (meas.last_lio_update_time - m.lio_time) * 1000.0f;
        meas.pcl_proc_next->points.push_back(pt);
        meas.pcl_proc_cur->points.pop_back();
      }

      std::reverse(meas.pcl_proc_next->points.begin(),
                   meas.pcl_proc_next->points.end());

      std::cout << std::fixed << std::setprecision(6)
                << "[pcl_proc_cur Timestamp Range] min: "
                << (double)meas.pcl_proc_cur->points.front().curvature /
                           (double)1000.0 +
                       meas.last_lio_update_time
                << ", max: "
                << (double)meas.pcl_proc_cur->points.back().curvature /
                           (double)1000.0 +
                       meas.last_lio_update_time
                << std::endl;

      std::cout << meas.pcl_proc_cur->points.back().curvature << " < "
                << (m.lio_time - meas.last_lio_update_time) * 1000.0f
                << std::endl;
      assert(meas.pcl_proc_cur->points.back().curvature <=
             (m.lio_time - meas.last_lio_update_time) * 1000.0f);

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;

      return true;
    }
    case LIO: {

      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      struct MeasureGroup m;
      // m.lio_time = meas.last_lio_update_time;
      m.vio_time = meas.last_lio_update_time;
      m.vio_time = meas.last_lio_update_time;

      // std::cout << "candid num: " << m_candidates.size() << std::endl;

      for (auto const &[cam_idx, val] : m_candidates) {
        m.imgs.push_back(val.second); // cv::Mat
        m.img_camera_indices.push_back(cam_idx);
      }

      // m.img = img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false;
      // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }
    default: {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
    }
    // return false;
    break;
  }

  case ONLY_LO: {
    if (!lidar_pushed) {
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())
        return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar
      meas.lidar_frame_beg_time =
          lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time =
          meas.lidar_frame_beg_time +
          meas.lidar->points.back().curvature / double(1000);
      lidar_pushed = true;
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO;
    // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default: {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage,
                                VIOManagerPtr vio_manager, int cam_idx) {
  cv::Mat img_rgb = vio_manager->img_cps[cam_idx];
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(bool publish_frame,
                                    const ros::Publisher &pubLaserCloudFullRes,
                                    VIOManagerPtr vio_manager, int cam_idx) {
  if (pcl_w_wait_pub->empty())
    return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en) {
    if (publish_frame) {
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgbs[cam_idx];
      for (size_t i = 0; i < size; i++) {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y,
                pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w));

        // TODO
        if (pf[2] < 0)
          continue;

        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(),
                                                     3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] *
          // inv_expo; pointRGB.b = pixel[0] * inv_expo; if (pointRGB.r > 255)
          // pointRGB.r = 255; else if (pointRGB.r < 0) pointRGB.r = 0; if
          // (pointRGB.g > 255) pointRGB.g = 255; else if (pointRGB.g < 0)
          // pointRGB.g = 0; if (pointRGB.b > 255) pointRGB.b = 255; else if
          // (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points)
            laserCloudWorldRGB->push_back(pointRGB);
        }
      }
      std::cout << "[CLOUD_REGISTERED] cam_idx: " << cam_idx << std::endl;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en) {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  } else {
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  }
  laserCloudmsg.header.stamp =
      ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    if (img_en) {
      *pcl_wait_save += *laserCloudWorldRGB;
    } else {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) &&
        pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") +
                            to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en) {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en) {
          pcd_writer.writeBinary(
              all_points_dir,
              *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir,
          // *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        } else {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " "
                     << _state.pos_end[2] << " " << q.w() << " " << q.x() << " "
                     << q.y() << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap) {
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size();
  if (size == 0)
    return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(
    const ros::Publisher &pubLaserCloudEffect,
    const std::vector<PointToPlane> &ptpl_list) {
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++) {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out) {
  const auto &state = show_imu_path ? last_IMU_state : _state;
  out.position.x = state.pos_end(0);
  out.position.y = state.pos_end(1);
  out.position.z = state.pos_end(2);

  auto temp_euler_cur = RotMtoEuler(state.rot_end);
  auto temp_geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
      temp_euler_cur(0), temp_euler_cur(1), temp_euler_cur(2));
  out.orientation.x = temp_geoQuat.x;
  out.orientation.y = temp_geoQuat.y;
  out.orientation.z = temp_geoQuat.z;
  out.orientation.w = temp_geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped,
                                 StatesGroup &state) {

  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp =
      ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(
      tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));

  auto temp_euler_cur = RotMtoEuler(state.rot_end);
  auto temp_geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
      temp_euler_cur(0), temp_euler_cur(1), temp_euler_cur(2));
  q.setW(temp_geoQuat.w);
  q.setX(temp_geoQuat.x);
  q.setY(temp_geoQuat.y);
  q.setZ(temp_geoQuat.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp,
                                        "camera_init", "aft_mapped"));

  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher) {
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time(LidarMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

void LIVMapper::save_path_to_file(const std::string &file_path) {
  if (path.poses.empty()) {
    ROS_WARN_ONCE("Path is empty. Nothing to write yet.");
    return;
  }

  static bool is_file_initialized = false;

  // --- 1. TUM Trajectory Format 파일 저장 ---
  const std::string &tum_output_path = file_path;

  // 파일 스트림 객체
  std::ofstream tum_file;

  if (!is_file_initialized) {
    tum_file.open(tum_output_path, std::ios::out);
    if (!tum_file.is_open()) {
      ROS_ERROR(
          "Failed to open file to save TUM trajectory for the first time: %s",
          tum_output_path.c_str());
      return;
    }
    // 파일 헤더 주석 (TUM 포맷 표준)
    tum_file << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    is_file_initialized = true;
  } else {
    // 두 번째 호출부터: 기존 파일 끝에 데이터를 추가(append)
    tum_file.open(tum_output_path, std::ios::app);
    if (!tum_file.is_open()) {
      ROS_ERROR("Failed to open file to append TUM trajectory: %s",
                tum_output_path.c_str());
      return;
    }
  }

  const auto &last_pose_stamped = path.poses.back();

  tum_file << std::fixed << std::setprecision(6)
           << last_pose_stamped.header.stamp.toSec() << " "
           << last_pose_stamped.pose.position.x << " "
           << last_pose_stamped.pose.position.y << " "
           << last_pose_stamped.pose.position.z << " "
           << last_pose_stamped.pose.orientation.x << " "
           << last_pose_stamped.pose.orientation.y << " "
           << last_pose_stamped.pose.orientation.z << " "
           << last_pose_stamped.pose.orientation.w << std::endl;

  tum_file.close();

  // ROS_INFO는 너무 자주 출력되므로 주석 처리하거나 ROS_INFO_THROTTLE 사용을
  // 권장 ROS_INFO("Successfully appended trajectory to %s",
  // tum_output_path.c_str());
}
