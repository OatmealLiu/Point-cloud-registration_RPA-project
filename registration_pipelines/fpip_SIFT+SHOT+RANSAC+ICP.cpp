/*
* Pipepline: SIFT --> SHOT --> RANSAC --> ICP [tested]
* Author: Team SLAMer, University of Trento
* Supervisor: MARIOLINO DE CECCO, ALESSANDRO LUCHETTI
* Ref:
*   [1]
*   [2]
*/

#include "pipeline_setup.h"

// Basics
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <pcl/console/time.h>

// Feature Extraction
#include <pcl/filters/filter.h>
#include <pcl/keypoints/sift_keypoint.h>


// Feature Descriptors
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/search/kdtree.h>

// Registration
#include <pcl/registration/correspondence_estimation.h>
#include <boost/thread/thread.hpp>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>

// Visualization
#include <pcl/visualization/pcl_visualizer.h>

using pcl::NormalEstimation;
using pcl::search::KdTree;
using namespace datastream;

// typedef shortcut
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

typedef pcl::SHOT352 SignT;
typedef pcl::PointCloud<SignT> PointCloudDesc;
typedef pcl::ReferenceFrame PointRef;

// Registration Visualization
void visualize_registration(PointCloud::Ptr pcd_src, PointCloud::Ptr pcd_tar, PointCloud::Ptr pcd_final)
{
    //int vp_1, vp_2;

    // Create a PCLVisualizer object
    pcl::visualization::PCLVisualizer viewer("Registration Viewer");

    //viewer.createViewPort (0.0, 0, 0.5, 1.0, vp_1);
   // viewer.createViewPort (0.5, 0, 1.0, 1.0, vp_2);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h(pcd_src, visualparm::srcRGBparm[0],
        visualparm::srcRGBparm[1],
        visualparm::srcRGBparm[2]);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> tar_h(pcd_tar, visualparm::tarRGBparm[0],
        visualparm::tarRGBparm[1],
        visualparm::tarRGBparm[2]);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> final_h(pcd_final, visualparm::finalRGBparm[0],
        visualparm::finalRGBparm[1],
        visualparm::finalRGBparm[2]);

    viewer.addPointCloud(pcd_src, src_h, "source cloud");
    viewer.addPointCloud(pcd_tar, tar_h, "tgt cloud");
    viewer.addPointCloud(pcd_final, final_h, "final cloud");
    //viewer.addCoordinateSystem(1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void visualize_keypoints(PointCloud::Ptr pcd_src_og,
    PointCloud::Ptr pcd_keypoints)
{
    pcl::visualization::PCLVisualizer viewer("Sift keypoint");
    viewer.setBackgroundColor(0, 0, 0);

    viewer.addPointCloud(pcd_src_og, "OG point cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "OG point cloud");

    viewer.addPointCloud(pcd_keypoints, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "Keypoints");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        boost::this_thread::sleep(boost::posix_time::microseconds(300));
    }
}

double computeCloudResolution(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (!pcl_isfinite((*cloud)[i].x))
        {
            continue;
        }

        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt(sqr_distances[1]);
            ++n_points;
        }
    }

    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

int main()
{
    std::cout << "------> Start ... Run Miumiu Run ..." << std::endl;
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Load point cloud data
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // create xyzrgb.pcd containers and load point cloud data to them
    PointCloud::Ptr rCloud_src(new PointCloud);
    PointCloud::Ptr rCloud_tar(new PointCloud);

    if (pcl::io::loadPCDFile<PointT>(filename_dense_src, *rCloud_src) == -1)
    {
        PCL_ERROR("Cannot load source.pcd file\n");
        return -1;
    }
    if (pcl::io::loadPCDFile<PointT>(filename_dense_tar, *rCloud_tar) == -1)
    {
        PCL_ERROR("Cannot load target.pcd file\n");
        return -1;
    }
    std::cout << "------> PCD Loaded ... " << std::endl;

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Point Cloud cleaning and Smooth surface normal estimation
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Remove NaN cloud points
    PointCloud::Ptr iCloud_src(new PointCloud);
    PointCloud::Ptr iCloud_tar(new PointCloud);

    std::vector<int> indices_src;
    pcl::removeNaNFromPointCloud(*rCloud_src, *iCloud_src, indices_src);
    pcl::io::savePCDFileASCII(filename_noNaN_src, *iCloud_src);
    std::cout << "------> Remove NaN points from *rCloud_src to *iCloud_src ...[saved]" << std::endl;

    std::vector<int> indices_tar;
    pcl::removeNaNFromPointCloud(*rCloud_tar, *iCloud_tar, indices_tar);
    pcl::io::savePCDFileASCII(filename_noNaN_tar, *iCloud_tar);
    std::cout << "------> Remove NaN points from *rCloud_tar to *iCloud_tar ...[saved]" << std::endl;

    pcl::search::KdTree<PointT>::Ptr modelKdTree_src(new pcl::search::KdTree<PointT>);
    modelKdTree_src->setInputCloud(iCloud_src);
    pcl::search::KdTree<PointT>::Ptr modelKdTree_tar(new pcl::search::KdTree<PointT>);
    modelKdTree_tar->setInputCloud(iCloud_tar);

    clock_t tik_normals_a = clock();
    
    // Calc normals for iCloud_src and iCloud_tar
    pcl::NormalEstimation<PointT, pcl::Normal> ne_iCloud_src;
    ne_iCloud_src.setSearchMethod(modelKdTree_src);
    ne_iCloud_src.setRadiusSearch(normalparm::radius_ic_dense);
    ne_iCloud_src.setInputCloud(iCloud_src);
    pcl::PointCloud<pcl::Normal>::Ptr iCloud_src_normals(new pcl::PointCloud<pcl::Normal>);
    ne_iCloud_src.compute(*iCloud_src_normals);
    std::cout << "------> Got *iCloud_src_normals ..." << std::endl;

    pcl::NormalEstimation<PointT, pcl::Normal> ne_iCloud_tar;
    ne_iCloud_tar.setSearchMethod(modelKdTree_tar);
    ne_iCloud_tar.setRadiusSearch(normalparm::radius_ic_dense);
    ne_iCloud_tar.setInputCloud(iCloud_tar);
    pcl::PointCloud<pcl::Normal>::Ptr iCloud_tar_normals(new pcl::PointCloud<pcl::Normal>);
    ne_iCloud_tar.compute(*iCloud_tar_normals);
    std::cout << "------> Got *iCloud_tar_normals ..." << std::endl;

    pcl::io::savePCDFileASCII(filename_normal_src, *iCloud_src_normals);
    pcl::io::savePCDFileASCII(filename_normal_tar, *iCloud_tar_normals);
    
    /* Read computed normals to boost the test
    pcl::PointCloud<pcl::Normal>::Ptr iCloud_src_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr iCloud_tar_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::io::loadPCDFile("boost_src_normals.pcd", *iCloud_src_normals);
    pcl::io::loadPCDFile("boost_tar_normals.pcd", *iCloud_tar_normals);
    pcl::io::savePCDFileASCII(filename_normal_src, *iCloud_src_normals);
    pcl::io::savePCDFileASCII(filename_normal_tar, *iCloud_tar_normals);
    std::cout << "------> Got *iCloud_src_normals ..." << std::endl;
    std::cout << "------> Got *iCloud_tar_normals ..." << std::endl;
    */
    clock_t tik_normals_b = clock();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Feature Extraction: run SIFT
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // create SIFT object
    // SIFT SRC
    clock_t tik_detect_a = clock();

    pcl::SIFTKeypoint<PointT, PointT> sift_src;
    sift_src.setInputCloud(iCloud_src);
    pcl::search::KdTree<PointT>::Ptr treeSiftSrc(new pcl::search::KdTree<PointT>());
    sift_src.setSearchMethod(treeSiftSrc);
    sift_src.setScales(siftparm::min_scale, siftparm::n_octaves, siftparm::n_scales_per_octave);
    sift_src.setMinimumContrast(siftparm::min_contrast);
    PointCloud::Ptr iKeypoints_src(new PointCloud);
    sift_src.compute(*iKeypoints_src);

    // SIFT TAR
    pcl::SIFTKeypoint<PointT, PointT> sift_tar;
    sift_tar.setInputCloud(iCloud_tar);
    pcl::search::KdTree<PointT>::Ptr treeSiftTar(new pcl::search::KdTree<PointT>());
    sift_tar.setSearchMethod(treeSiftTar);
    sift_tar.setScales(siftparm::min_scale, siftparm::n_octaves, siftparm::n_scales_per_octave);
    sift_tar.setMinimumContrast(siftparm::min_contrast);
    PointCloud::Ptr iKeypoints_tar(new PointCloud);
    sift_tar.compute(*iKeypoints_tar);

    clock_t tik_detect_b = clock();

    pcl::io::savePCDFileASCII(filename_sift_src, *iKeypoints_src);
    std::cout << "------> SIFTed: size *iCloud_src to *iKeypoints_src from [ " << iCloud_src->size() << " -> " << iKeypoints_src->size() << " ]"
        << " ... [saved]" << std::endl;

    pcl::io::savePCDFileASCII(filename_sift_tar, *iKeypoints_tar);
    std::cout << "------> SIFTed: size *iCloud_tar to *iKeypoints_tar from [ " << iCloud_tar->size() << " -> " << iKeypoints_tar->size() << " ]"
        << " ... [saved]" << std::endl;

    visualize_keypoints(iCloud_src, iKeypoints_src);

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Feature Descriptor: compute SHOT feature descriptors
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    clock_t tik_descriptors_a = clock();

    // Calc SHOT feature descriptors
    pcl::SHOTEstimation<PointT, pcl::Normal, SignT> shot_src;
    pcl::search::KdTree<PointT>::Ptr tree_shot_src(new pcl::search::KdTree<PointT>);
    shot_src.setSearchMethod(tree_shot_src);
    shot_src.setInputCloud(iKeypoints_src);
    shot_src.setInputNormals(iCloud_src_normals);
    shot_src.setRadiusSearch(shotparm::radius_dense);
    shot_src.setSearchSurface(iCloud_src);
    PointCloudDesc::Ptr rSHOTDesc_src(new PointCloudDesc());
    shot_src.compute(*rSHOTDesc_src);
    std::cout << "------> Computed *rSHOTDesc_src SHOT descriptor values ..." << std::endl;
    std::cout << rSHOTDesc_src->is_dense << std::endl;
    std::cout << rSHOTDesc_src->size() << std::endl;

    pcl::SHOTEstimation<PointT, pcl::Normal, SignT> shot_tar;
    pcl::search::KdTree<PointT>::Ptr tree_shot_tar(new pcl::search::KdTree<PointT>);
    shot_tar.setSearchMethod(tree_shot_tar);
    shot_tar.setInputCloud(iKeypoints_tar);
    shot_tar.setInputNormals(iCloud_tar_normals);
    shot_tar.setRadiusSearch(shotparm::radius_dense);
    shot_tar.setSearchSurface(iCloud_tar);
    PointCloudDesc::Ptr rSHOTDesc_tar(new PointCloudDesc());
    shot_tar.compute(*rSHOTDesc_tar);
    std::cout << "------> Computed *rSHOTDesc_tar SHOT descriptor values ..." << std::endl;
    std::cout << rSHOTDesc_tar->is_dense << std::endl;
    std::cout << rSHOTDesc_tar->size() << std::endl;

    clock_t tik_descriptors_b = clock();

    pcl::io::savePCDFileASCII(filename_desc_shot_src, *rSHOTDesc_src);
    pcl::io::savePCDFileASCII(filename_desc_shot_tar, *rSHOTDesc_tar);

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Coarse Registration: RANSAC
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // RANSAC  Coarse Regisration
    clock_t tik_ransac_a = clock();

    pcl::SampleConsensusInitialAlignment<PointT, PointT, SignT> scia;
    PointCloud::Ptr sac_result(new PointCloud);
    // RANSAC setup
    scia.setInputSource(iKeypoints_src);
    scia.setInputTarget(iKeypoints_tar);
    scia.setSourceFeatures(rSHOTDesc_src);
    scia.setTargetFeatures(rSHOTDesc_tar);
    scia.setMaximumIterations(sacparm::max_iterations);
    scia.setNumberOfSamples(sacparm::num_sampels);
    scia.setCorrespondenceRandomness(sacparm::corresp_randomness);
    //scia.setMaxCorrespondenceDistance(sacparm::max_corresp_dist);
    //scia.setInlierFraction(sacparm::inlier_fraction);
    //scia.setSimilarityThreshold(sacparm::similarity_threshold)

    scia.align(*sac_result);
    std::cout << "------> RANSAC has converged:" << scia.hasConverged() << "  Score: " << scia.getFitnessScore() << std::endl;

    Eigen::Matrix4f sac_rotTrans;
    sac_rotTrans = scia.getFinalTransformation();
    std::cout << "------> RANSAC Rot-Translation Matrix:" << std::endl;
    std::cout << sac_rotTrans << std::endl;

    clock_t tik_ransac_b = clock();

    pcl::io::savePCDFileASCII(filename_SAC_trans_src, *sac_result);

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Fine Registration: ICP
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    clock_t tik_icp_a = clock();

    // ICP regisrater
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    PointCloud::Ptr icp_result(new PointCloud);

    // ICP setup
    icp.setInputSource(iKeypoints_src);
    icp.setInputTarget(iCloud_tar);

    icp.setMaxCorrespondenceDistance(icpparm::max_corresp_dist);
    icp.setMaximumIterations(icpparm::max_iterations);
    icp.setTransformationEpsilon(icpparm::trans_epsilon);
    icp.setEuclideanFitnessEpsilon(icpparm::euclidean_fitness_epsilon);
    icp.align(*icp_result, sac_rotTrans);

    std::cout << "------> ICP has converged:" << icp.hasConverged() << "  Score: " << icp.getFitnessScore() << std::endl;

    clock_t tik_icp_b = clock();

    Eigen::Matrix4f icp_rotTrans;
    icp_rotTrans = icp.getFinalTransformation();
    std::cout << "------> ICP Rot-Translation Matrix:" << std::endl;
    std::cout << icp_rotTrans << std::endl;

    pcl::transformPointCloud(*iCloud_src, *icp_result, icp_rotTrans);
    pcl::io::savePCDFileASCII(filename_SAC_ICP_trans_src, *icp_result);

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Visualization
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    visualize_registration(iCloud_src, iCloud_tar, icp_result);

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Pipeline Evaluation
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Evaluate Time Cost
    std::cout << "------> Total time     : " << (double)(tik_icp_b - tik_normals_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "------> Normals time   : " << (double)(tik_normals_b - tik_normals_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "------> SIFT time      : " << (double)(tik_detect_b - tik_detect_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "------> SHOT time      : " << (double)(tik_descriptors_b - tik_descriptors_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "------> RANSAC time    : " << (double)(tik_ransac_b - tik_ransac_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    std::cout << "------> ICP time       : " << (double)(tik_icp_b - tik_icp_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;

    // Evaluate RMSE
    pcl::registration::CorrespondenceEstimation<PointT, PointT> corre;
    corre.setInputSource(icp_result);
    corre.setInputTarget(iCloud_tar);

    pcl::Correspondences all_corre;

    corre.determineReciprocalCorrespondences(all_corre);

    float sum = 0.0, sum_x = 0.0, sum_y = 0.0, sum_z = 0.0, rmse, rmse_x, rmse_y, rmse_z;

    std::vector<float> co;

    for (size_t j = 0; j < all_corre.size(); j++) {
        sum += all_corre[j].distance;
        co.push_back(all_corre[j].distance);
        sum_x += pow((iCloud_tar->points[all_corre[j].index_match].x - icp_result->points[all_corre[j].index_query].x), 2);
        sum_y += pow((iCloud_tar->points[all_corre[j].index_match].y - icp_result->points[all_corre[j].index_query].y), 2);
        sum_z += pow((iCloud_tar->points[all_corre[j].index_match].z - icp_result->points[all_corre[j].index_query].z), 2);
    }

    rmse = sqrt(sum / all_corre.size());        // Total RMSE on XYZ
    rmse_x = sqrt(sum_x / all_corre.size());    // RMSE on X-axis
    rmse_y = sqrt(sum_y / all_corre.size());    // RMSE on X-axis
    rmse_z = sqrt(sum_z / all_corre.size());    // RMSE on X-axis

    std::vector<float>::iterator max = max_element(co.begin(), co.end());       // Get the Farthest cloud point pair
    std::vector<float>::iterator min = min_element(co.begin(), co.end());       // Get the Nearest cloud point pair

    std::cout << "------> Number of correspondent points: " << all_corre.size() << std::endl;
    std::cout << "------> Maximum distance: " << sqrt(*max) * 100 << "cm" << std::endl;
    std::cout << "------> Minimum distance: " << sqrt(*min) * 100 << "cm" << std::endl;

    std::cout << "------> Total RMSE     :" << rmse << " m" << std::endl;
    std::cout << "------> RMSE on x-axis :" << rmse_x << " m" << std::endl;
    std::cout << "------> RMSE on y-axis :" << rmse_y << " m" << std::endl;
    std::cout << "------> RMSE on z-axis :" << rmse_z << " m" << std::endl;

    // Record the pipeline evaluation
    ofstream fout(datastream::filename_pip_shot);
    fout << "Pipeline Evaluation Result\n";
    fout << "Purpose             : 3D XYZRGB Point Cloud Registration\n";
    fout << "Keypoints Detector  : SIFT\n";
    fout << "Featrue Descriptor  : SHOT\n";
    fout << "Coarse Registration : RANSAC\n";
    fout << "Fina Registration   : ICP\n";
    fout << "Author              : @Team SLAMer\n";
    fout << "Supervisor          : MARIOLINO DE CECCO, ALESSANDRO LUCHETTI\n";

    fout << "\nTime Evaluation\n";
    fout << "------> Total time     : " << (double)(tik_icp_b - tik_normals_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    fout << "------> Normals time   : " << (double)(tik_normals_b - tik_normals_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    fout << "------> SIFT time      : " << (double)(tik_detect_b - tik_detect_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    fout << "------> SHOT time      : " << (double)(tik_descriptors_b - tik_descriptors_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    fout << "------> RANSAC time    : " << (double)(tik_ransac_b - tik_ransac_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;
    fout << "------> ICP time       : " << (double)(tik_icp_b - tik_icp_a) / (double)CLOCKS_PER_SEC << " s" << std::endl;

    fout << "\nRMSE Error Evaluation\n";
    fout << "------> Number of correspondent points: " << all_corre.size() << std::endl;
    fout << "------> Maximum distance: " << sqrt(*max) * 100 << "cm" << std::endl;
    fout << "------> Minimum distance: " << sqrt(*min) * 100 << "cm" << std::endl;
    fout << "------> Total RMSE     :" << rmse << " m" << std::endl;
    fout << "------> RMSE on x-axis :" << rmse_x << " m" << std::endl;
    fout << "------> RMSE on y-axis :" << rmse_y << " m" << std::endl;
    fout << "------> RMSE on z-axis :" << rmse_z << " m" << std::endl;

    fout.close();

    std::cout << "------> Saved evaluation result '" << datastream::filename_pip_shot << "', run run Miumiu run...\n";

    return (0);
}