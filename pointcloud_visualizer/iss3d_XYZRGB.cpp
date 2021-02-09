/*
* Pipepline: SIFT --> SHOTColor --> RANSAC --> ICP
*/

#include "pipeline_setup.h"

// Basics
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>

// Feature Extraction
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/sift_keypoint.h>

// Feature Descriptors
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/board.h>       
#include <pcl/features/boundary.h>

// Registration
#include <pcl/registration/correspondence_estimation.h>
#include <boost/thread/thread.hpp>
#include <pcl/console/time.h>
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
typedef pcl::SHOT1344 SignT;
typedef pcl::PointCloud<SignT> PointCloudDesc;
typedef pcl::ReferenceFrame PointRef;

// Registration Visualization
void visualize_pcd(PointCloud::Ptr pcd_src, PointCloud::Ptr pcd_tar, PointCloud::Ptr pcd_final)
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

//由旋转平移矩阵计算旋转角度
void matrix2angle(Eigen::Matrix4f& result_trans, Eigen::Vector3f& result_angle)
{
    double ax, ay, az;
    if (result_trans(2, 0) == 1 || result_trans(2, 0) == -1)
    {
        az = 0;
        double dlta;
        dlta = atan2(result_trans(0, 1), result_trans(0, 2));
        if (result_trans(2, 0) == -1)
        {
            ay = M_PI / 2;
            ax = az + dlta;
        }
        else
        {
            ay = -M_PI / 2;
            ax = -az + dlta;
        }
    }
    else
    {
        ay = -asin(result_trans(2, 0));
        ax = atan2(result_trans(2, 1) / cos(ay), result_trans(2, 2) / cos(ay));
        az = atan2(result_trans(1, 0) / cos(ay), result_trans(0, 0) / cos(ay));
    }
    result_angle << ax, ay, az;
}

void eraseInValidPoints(PointCloud::Ptr ini_cloud)
{
    PointCloud::iterator it = ini_cloud->points.begin();
    while (it != ini_cloud->points.end())
    {
        float x, y, z, rgb;
        x = it->x;
        y = it->y;
        z = it->z;
        //rgb = it->rgb;
        //cout << "x: " << x << "  y: " << y << "  z: " << z << "  rgb: " << rgb << endl;
        if (!pcl_isfinite(x) || !pcl_isfinite(y) || !pcl_isfinite(z) || !pcl_isfinite(rgb))
        {
            it = ini_cloud->points.erase(it);
        }
        else
            ++it;
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
    tree.setInputCloud(cloud);   //设置输入点云

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (!pcl_isfinite((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        //运算第二个临近点，因为第一个点是它本身
        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);//return :number of neighbors found 
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

    if (pcl::io::loadPCDFile<PointT>("WandC_src.pcd", *rCloud_src) == -1)
    {
        PCL_ERROR("Cannot load source.pcd file\n");
        return -1;
    }
    if (pcl::io::loadPCDFile<PointT>("WandC_tar.pcd", *rCloud_tar) == -1)
    {
        PCL_ERROR("Cannot load target.pcd file\n");
        return -1;
    }
    std::cout << "-----> PCD Loaded ... " << std::endl;

    clock_t start = clock();

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Feature Extraction: run SIFT
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Remove NaN cloud points
    PointCloud::Ptr iCloud_src(new PointCloud);
    PointCloud::Ptr iCloud_tar(new PointCloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr wCloud_src(new  pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr wCloud_tar(new  pcl::PointCloud<pcl::PointXYZ>);

    std::vector<int> indices_src;
    pcl::removeNaNFromPointCloud(*rCloud_src, *iCloud_src, indices_src);
    pcl::io::savePCDFileASCII(filename_noNaN_src, *iCloud_src);
    std::cout << "------> Remove NaN points from *rCloud_src to *iCloud_src ...[saved]" << std::endl;
    pcl::copyPointCloud(*iCloud_src, *wCloud_src);



    std::vector<int> indices_tar;
    pcl::removeNaNFromPointCloud(*rCloud_tar, *iCloud_tar, indices_tar);
    pcl::io::savePCDFileASCII(filename_noNaN_tar, *iCloud_tar);
    std::cout << "------> Remove NaN points from *rCloud_tar to *iCloud_tar ...[saved]" << std::endl;
    pcl::copyPointCloud(*iCloud_tar, *wCloud_tar);

    // create ISS object
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det_src;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_iss_src(new pcl::search::KdTree<pcl::PointXYZ>());
    iss_det_src.setInputCloud(wCloud_src);
    iss_det_src.setSearchMethod(tree_iss_src);
    double iResolution_src = computeCloudResolution(iCloud_tar);
    iss_det_src.setSalientRadius(6 * iResolution_src);
    iss_det_src.setNonMaxRadius(4 * iResolution_src);
    iss_det_src.setNormalRadius(6 * iResolution_src);
    iss_det_src.setBorderRadius(4 * iResolution_src);
    iss_det_src.setAngleThreshold(static_cast<float> (M_PI) / 2.0);
    iss_det_src.setMinNeighbors(iss3dparm::minKnn);
    iss_det_src.setThreshold21(iss3dparm::threshold21);
    iss_det_src.setThreshold32(iss3dparm::threshold32);
    pcl::PointCloud<pcl::PointXYZ>::Ptr rKeypoints_src(new  pcl::PointCloud<pcl::PointXYZ>);
    iss_det_src.compute(*rKeypoints_src);

    PointCloud::Ptr iKeypoints_src(new PointCloud);
    pcl::PointIndices idx_kp_src =*iss_det_src.getKeypointsIndices();
    std::cout << "Ptr:" << *iss_det_src.getKeypointsIndices() << std::endl;
    std::cout << idx_kp_src << endl;

    pcl::copyPointCloud(*iCloud_src, idx_kp_src, *iKeypoints_src);


    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_det_tar;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_iss_tar(new pcl::search::KdTree<pcl::PointXYZ>());
    iss_det_tar.setInputCloud(wCloud_tar);
    iss_det_tar.setSearchMethod(tree_iss_tar);
    double iResolution_tar = computeCloudResolution(iCloud_tar);
    iss_det_tar.setSalientRadius(6 * iResolution_tar);
    iss_det_tar.setNonMaxRadius(4 * iResolution_tar);
    iss_det_tar.setNormalRadius(6 * iResolution_tar);
    iss_det_tar.setBorderRadius(4 * iResolution_tar);
    iss_det_tar.setAngleThreshold(static_cast<float> (M_PI) / 2.0);
    iss_det_tar.setMinNeighbors(iss3dparm::minKnn);
    iss_det_tar.setThreshold21(iss3dparm::threshold21);
    iss_det_tar.setThreshold32(iss3dparm::threshold32);
    pcl::PointCloud<pcl::PointXYZ>::Ptr rKeypoints_tar(new  pcl::PointCloud<pcl::PointXYZ>);
    iss_det_tar.compute(*rKeypoints_tar);

    PointCloud::Ptr iKeypoints_tar(new PointCloud);
    pcl::PointIndices idx_kp_tar = *iss_det_tar.getKeypointsIndices();
    std::cout << "Ptr:" << *iss_det_tar.getKeypointsIndices() << std::endl;
    std::cout << idx_kp_tar << endl;

    pcl::copyPointCloud(*iCloud_tar, idx_kp_tar, *iKeypoints_tar);

    pcl::io::savePCDFileASCII(filename_iss3d_src, *iKeypoints_src);
    std::cout << "ISS3Ding: size *iCloud_src to *iKeypoints_src from [ " << iCloud_src->size() << " -> " << iKeypoints_src->size() << " ]"
        << " ... [saved]" << std::endl;

    pcl::io::savePCDFileASCII(filename_iss3d_tar, *iKeypoints_tar);
    std::cout << "ISS3Ding: size *iCloud_tar to *iKeypoints_tar from [ " << iCloud_tar->size() << " -> " << iKeypoints_tar->size() << " ]"
        << " ... [saved]" << std::endl;


    //可视化输入点云和关键点
    pcl::visualization::PCLVisualizer viewer("Sift keypoint");
    viewer.setBackgroundColor(0, 0, 0);

    viewer.addPointCloud(iCloud_src, "OG point cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "OG point cloud");

    viewer.addPointCloud(iKeypoints_src, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "Keypoints");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        boost::this_thread::sleep(boost::posix_time::microseconds(1000));
    }
    return 0;
}