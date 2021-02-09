#include <iostream>
#include <time.h>

// Basics
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>

// Feature Extraction
#include <pcl/keypoints/sift_keypoint.h>

// Feature Descriptors
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>

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

int main(int argc, char* argv[])
{
    std::string filename = "pointcloud2_bigrainbow.pcd";
    std::string rltFilename = "pointcloud2_bigrainbow_sift.pcd";

    //Create a pointer of PointCloud<pcl::PointXYZRGB> object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZ>);

    // open PointXYZRGB .pcd file
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud_xyzrgb) == -1)
    {
        PCL_ERROR("Couldn't read file pointcloud_rgb.pcd \n");
        return (-1);
    }

    //去除NAN点
    std::vector<int> indices_src; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_xyzrgb, *cloud_xyzrgb, indices_src);
    std::cout << "------> Remove *cloud_xyzrgb nan points..." << std::endl;
    pcl::io::savePCDFileASCII("pointcloud1_noNaN_src.pcd", *cloud_xyzrgb);


    const float min_scale = 0.1f;                                                                   //设置尺度空间中最小尺度的标准偏差          
    const int n_octaves = 6;                                                                        //设置高斯金字塔组（octave）的数目            
    const int n_scales_per_octave = 4;                                                              //设置每组（octave）计算的尺度  
    const float min_contrast = 0.1f;                                                                //设置限制关键点检测的阈值       

    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointXYZ> sift;                                  //创建sift关键点检测对象

    pcl::PointCloud<pcl::PointXYZ> result;
    sift.setInputCloud(cloud_xyzrgb);                                                               //设置输入点云

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    sift.setSearchMethod(tree);                                                                     //创建一个空的kd树对象tree，并把它传递给sift检测对象
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);                                      //指定搜索关键点的尺度范围
    sift.setMinimumContrast(min_contrast);                                                          //设置限制关键点检测的阈值
    sift.compute(result);                                                                           //执行sift关键点检测，保存结果在result



    //将点类型pcl::PointWithScale的数据转换为点类型pcl::PointXYZRGB的数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(result, *cloud_temp);

    // Save the result point cloud after SIFT
    std::cout << "------> Sifted saving......" << std::endl;
    pcl::io::savePCDFileASCII<pcl::PointXYZ>(rltFilename, *cloud_temp);

    //可视化输入点云和关键点
    pcl::visualization::PCLVisualizer viewer("Sift keypoint");
    viewer.setBackgroundColor(0, 0, 0);

    viewer.addPointCloud(cloud_xyzrgb, "OG point cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "OG point cloud");

    viewer.addPointCloud(cloud_temp, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255, "Keypoints");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        boost::this_thread::sleep(boost::posix_time::microseconds(1000));
    }
    return 0;

}