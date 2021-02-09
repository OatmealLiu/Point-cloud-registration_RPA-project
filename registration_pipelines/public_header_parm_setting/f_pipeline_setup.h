/*
* Parameter Setting
* Author: Team SLAMer, University of Trento
* Supervisor: MARIOLINO DE CECCO, ALESSANDRO LUCHETTI
* Ref:
*   [1]
*   [2]
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <vector>


namespace datastream
{
    // transXYZ(+6,0,0) rotXYZ(0,+30,0)
    //std::string filename_sparse_src = "pointcloud_sparse_src_InTheTorii.pcd";
    //std::string filename_sparse_tar = "pointcloud_sparse_tar_InTheTorii.pcd";

    // transXYZ(0,0,0) rotXYZ(0,+30,0)
    std::string filename_dense_src = "pointcloud_dense_src_WarriorInTheHouse.pcd";
    std::string filename_dense_tar = "pointcloud_dense_tar_WarriorInTheHouse.pcd";

    std::string filename_noNaN_src = "pointcloud_src_noNaN.pcd";
    std::string filename_noNaN_tar = "pointcloud_tar_noNaN.pcd";

    std::string filename_sift_src = "pointcloud_src_sift_keypoints.pcd";
    std::string filename_sift_tar = "pointcloud_tar_sift_keypoints.pcd";

    std::string filename_iss3d_src = "pointcloud_src_iss3d_keypoints.pcd";
    std::string filename_iss3d_tar = "pointcloud_tar_iss3d_keypoints.pcd";
    
    std::string filename_normal_src = "iCloud_src_normals.pcd";
    std::string filename_normal_tar = "iCloud_tar_normals.pcd";

    std::string filename_SAC_trans_src = "pointcloud_src_RANSAC_transformed.pcd";
    std::string filename_SAC_ICP_trans_src = "pointcloud_src_ICP_transformed_src.pcd";

    std::string filename_desc_cshot_src = "rCSHOTDesc_src.pcd";
    std::string filename_desc_shot_src = "rSHOTDesc_src.pcd";
    std::string filename_desc_si_src = "rSpinImageDesc_src.pcd";
    std::string filename_desc_fpfh_src = "rFPFHDesc_src.pcd";
    std::string filename_desc_sift_src = "rSIFTDesc_src.pcd";

    std::string filename_desc_cshot_tar = "rCSHOTDesc_tar.pcd";
    std::string filename_desc_shot_tar = "rSHOTDesc_tar.pcd";
    std::string filename_desc_si_tar = "rSpinImageDesc_tar.pcd";
    std::string filename_desc_fpfh_tar = "rFPFHDesc_tar.pcd";
    std::string filename_desc_sift_tar = "rSIFTDesc_tar.pcd";

    std::string filename_pip_cshot = "pip_evaluation_ISS3D+CSHOT+RANSAC+ICP.txt";
    std::string filename_pip_shot = "pip_evaluation_SIFT+SHOT+RANSAC+ICP.txt";
    std::string filename_pip_si = "pip_evaluation_SIFT+SpinImage+RANSAC+ICP.txt";
    std::string filename_pip_fpfh = "pip_evaluation_SIFT+FPFH+RANSAC+ICP.txt";
    std::string filename_pip_sift = "pip_evaluation_SIFT+RANSAC+ICP.txt";
}

// Normal Estimation
namespace normalparm
{
    // Warrior and Cat scene: r = 1.0 (no nan)
    //                        r = 0.5 no nan
    // 鸟居                   r = 
    double radius_ic_sparse = 1.0;

    // Small flat furniture scene:
    //      r = 0.3 no nan at all(ok, better)
    //      r = 0.2 (a little nan)(ok)
    double radius_ic_dense = 0.3;

    // r = 2.0~2.5(ok for si,2.5 no nan)
    double radius_kp_dense = 2.5;
    //int knn = 20;
}

// Feature Detectors
namespace iss3dparm
{
    int minKnn = 5;
    double threshold21 = 0.975;
    double threshold32 = 0.975;
    double salientRadius = 2.4;
    double nonMaxRadius = 1.6;
    double nonNormalRadius = 2.4;
    double nonBorderRadius = 1.6;
    //float angleThreshold = static_cast<float> (M_PI) / 2.0;
}
namespace siftparm
{
    const float min_scale = 0.1f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.1f;
}

// Feature Descriptors
namespace fpfhparm
{
    double radius_sparse = 3.0;

    // r = 0.2, no nan(ok)
    double radius_dense = 0.2;
}

namespace siparm
{
    unsigned int image_width = 8U;
    double support_ang_cos = 0.5;
    unsigned int min_pts_neighb = 16U;

    double radius_sparse = 3.0;

    // r = 1.0(ok for si)
    double radius_dense = 1.0;
}

namespace shotparm
{
    // Warrior and Cat scene: r = 1.0 (no nan)
    //                        r = 
    // 鸟居                   r = 2?
    double radius_sparse = 3.0;
    // Dense: 0.1(x), 0.2(x), 0.3(ok)
    double radius_dense = 0.3;
}
namespace cshotparm
{
    double radiusLRF = 0.2;

    // Warrior and Cat scene: r = 3.0 (no nan in desc)
    // Torii and Cat scene: r = 1.0 (no nan)
    double radius_sparse = 3.0;

    // Warrior in the house scene
    //  r = 0.1, no nan(ok)
    //  r = 0.15 no nan(ok)
    double radius_dense = 0.15;
}

namespace vfhparm
{
    double radius_sparse = 3.0;

    double radius_dense = 0.15;
}

// Registeration
namespace sacparm
{
    int max_iterations = 2000;
    int num_sampels = 3;
    int corresp_randomness = 20;
    float max_corresp_dist = 1.25f;
    float inlier_fraction = 0.25f;
    float similarity_threshold = 0.9f;
}

namespace icpparm
{
    float max_corresp_dist = 20.0f; // most important
    int max_iterations = 10000;
    double trans_epsilon = 1e-10;
    double euclidean_fitness_epsilon = 0.2;
}

// Point Cloud Visualization
namespace visualparm
{
    float srcRGBparm[] = { 255.0f, 255.0f, 255.0f };
    float tarRGBparm[] = { 255.0f, 0.0f, 0.0f };
    float finalRGBparm[] = { 0.0f, 0.0f, 255.0f };
}