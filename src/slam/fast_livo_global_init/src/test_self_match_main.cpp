// V3 검증: cloudGlobal.pcd를 source/target으로 넣어 KISS-Matcher 정합 동작 확인.
//   case 1: src = tgt 그대로  →  R≈I, t≈0
//   case 2: src = R·tgt + t (known)  →  복원된 R, t 비교
//
// KISSMatcher 입력은 Vector3f, 출력 RegistrationSolution은 Matrix3d/Vector3d.

#include <Eigen/Geometry>
#include <chrono>
#include <iostream>
#include <kiss_matcher/KISSMatcher.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

namespace {

std::vector<Eigen::Vector3f> toVec(const pcl::PointCloud<pcl::PointXYZ> &cloud) {
  std::vector<Eigen::Vector3f> out;
  out.reserve(cloud.size());
  for (const auto &p : cloud) out.emplace_back(p.x, p.y, p.z);
  return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &in, float leaf) {
  auto out = pcl::PointCloud<pcl::PointXYZ>::Ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(in);
  vg.setLeafSize(leaf, leaf, leaf);
  vg.filter(*out);
  return out;
}

void runCase(const std::string &name,
             const std::vector<Eigen::Vector3f> &src,
             const std::vector<Eigen::Vector3f> &tgt,
             const Eigen::Matrix3d &expected_R,
             const Eigen::Vector3d &expected_t,
             float resolution) {
  std::cout << "\n=== " << name << " ===\n";
  std::cout << "src=" << src.size() << " tgt=" << tgt.size() << "\n";

  kiss_matcher::KISSMatcherConfig cfg(resolution);
  kiss_matcher::KISSMatcher matcher(cfg);

  const auto t0 = std::chrono::steady_clock::now();
  const auto sol = matcher.estimate(src, tgt);
  const auto dt = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count();

  std::cout << "elapsed=" << dt << " s\n";
  std::cout << "valid=" << sol.valid << "\n";
  std::cout << "R=\n" << sol.rotation << "\n";
  std::cout << "t=" << sol.translation.transpose() << "\n";
  std::cout << "expected R=\n" << expected_R << "\n";
  std::cout << "expected t=" << expected_t.transpose() << "\n";

  const double t_err = (sol.translation - expected_t).norm();
  const double R_err = (sol.rotation - expected_R).norm();
  std::cout << "|t_err|=" << t_err << " m,  |R_err|_F=" << R_err << "\n";

  const bool pass = sol.valid && t_err < 1.0 && R_err < 0.5;
  std::cout << "[" << (pass ? "PASS" : "FAIL") << "] " << name << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <pcd_path> [voxel_leaf=0.3]\n";
    return 1;
  }
  const std::string pcd_path = argv[1];
  const float leaf = (argc >= 3) ? std::stof(argv[2]) : 0.3f;

  auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) != 0) {
    std::cerr << "failed to load " << pcd_path << "\n";
    return 2;
  }
  std::cout << "loaded " << cloud->size() << " points from " << pcd_path
            << "\n";

  auto tgt_pcl = voxel(cloud, leaf);
  std::cout << "voxel(" << leaf << ") -> " << tgt_pcl->size() << " points\n";
  const auto tgt_vec = toVec(*tgt_pcl);

  // -------- case 1: identity --------
  runCase("case1 identity", tgt_vec, tgt_vec,
          Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), leaf);

  // -------- case 2: known transform --------
  const double yaw = 0.4;  // ~23 deg
  Eigen::Matrix3d R =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d t(2.5, -1.2, 0.0);

  std::vector<Eigen::Vector3f> src_transformed;
  src_transformed.reserve(tgt_vec.size());
  for (const auto &p : tgt_vec) {
    const Eigen::Vector3d pd = p.cast<double>();
    const Eigen::Vector3d q = R * pd + t;
    src_transformed.emplace_back(q.cast<float>());
  }

  // KISS-Matcher가 tgt ≈ R_sol · src + t_sol을 푼다.
  // src = R·tgt + t  →  tgt = Rᵀ·src - Rᵀ·t
  const Eigen::Matrix3d expected_R = R.transpose();
  const Eigen::Vector3d expected_t = -R.transpose() * t;
  runCase("case2 known transform (yaw=0.4, t=(2.5,-1.2,0))",
          src_transformed, tgt_vec, expected_R, expected_t, leaf);

  return 0;
}
