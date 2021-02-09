#include <iostream>
#include <time.h>

// Basics
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/common/impl/io.hpp>



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

/*
using namespace pcl;
namespace pcl
{
    template<>
    struct SIFTKeypointFieldSelector<PointXYZRGB>
    {
        inline float
            operator () (const PointXYZRGB& p) const
        {
            return p.z;
        }
    };
}*/

int main(int argc, char* argv[])
{
    std::string filename = "pointcloud2_tar_rainbow.pcd";

    //Create a pointer of PointCloud<pcl::PointXYZRGB> object
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);

    // open PointXYZRGB .pcd file
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud_xyzrgb) == -1)
    {
        PCL_ERROR("Couldn't read file pointcloud_rgb.pcd \n");
        return (-1);
    }
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud_xyz) == -1)
    {
        PCL_ERROR("Couldn't read file pointcloud_rgb.pcd \n");
        return (-1);
    }

    //去除NAN点
    std::vector<int> indices_src_xyzrgb; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_xyzrgb, *cloud_xyzrgb, indices_src_xyzrgb);
    std::cout << "------> Remove *cloud_xyzrgb nan points..." << std::endl;
    pcl::io::savePCDFileASCII("pointcloud2_noNaN_src_XYZRGB.pcd", *cloud_xyzrgb);

    std::vector<int> indices_src_xyz; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_xyz, *cloud_xyz, indices_src_xyz);
    std::cout << "------> Remove *cloud_xyz nan points..." << std::endl;
    pcl::io::savePCDFileASCII("pointcloud2_noNaN_src_XYZ.pcd", *cloud_xyz);

    const float min_scale = 0.1f;                                                                   //设置尺度空间中最小尺度的标准偏差          
    const int n_octaves = 6;                                                                        //设置高斯金字塔组（octave）的数目            
    const int n_scales_per_octave = 4;                                                              //设置每组（octave）计算的尺度  
    const float min_contrast = 0.1f;                                                                //设置限制关键点检测的阈值       

    pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;                                  //创建sift关键点检测对象

    pcl::PointCloud<pcl::PointWithScale>::Ptr sifted_xyz_rlt(new pcl::PointCloud<pcl::PointWithScale>);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    sift.setSearchMethod(tree);                                                                     //创建一个空的kd树对象tree，并把它传递给sift检测对象
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);                                      //指定搜索关键点的尺度范围
    sift.setMinimumContrast(min_contrast);                                                          //设置限制关键点检测的阈值

    sift.setInputCloud(cloud_xyzrgb);                                                               //设置输入点云
    sift.compute(*sifted_xyz_rlt);                                                                   //执行sift关键点检测，保存结果在result
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sifted_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*sifted_xyz_rlt, *cloud_sifted_xyzrgb);
    //pcl::PointCloud<pcl::PointCloud<int>> siftedIndices;
    //pcl::PointIndices::ConstPtr ptrIdx(new pcl::PointIndices);

    std::cout << sift.getKeypointsIndices() << std::endl;
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr sifted_xyzRGB_rlt(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::copyPointCloud(*cloud_xyzrgb, *ptrIdx, *sifted_xyzRGB_rlt);
    
    // Save the result point cloud after SIFT
    std::cout << "------> Sifted saving......" << std::endl;
    pcl::io::savePCDFileASCII<pcl::PointXYZRGB>("pointcloud2_bigrainbow_sift_XYZ.pcd", *cloud_sifted_xyzrgb);
    //pcl::io::savePCDFileASCII<pcl::PointXYZRGB>("pointcloud2_bigrainbow_sift_XYZRGB", *sifted_xyzRGB_rlt);


    //可视化输入点云和关键点
    pcl::visualization::PCLVisualizer viewer("Sift keypoint");
    viewer.setBackgroundColor(0, 0, 0);

    viewer.addPointCloud(cloud_xyzrgb, "OG point cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "OG point cloud");

    viewer.addPointCloud(cloud_sifted_xyzrgb, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "Keypoints");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        boost::this_thread::sleep(boost::posix_time::microseconds(1000));
    }
    return 0;

}