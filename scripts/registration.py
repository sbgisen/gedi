#!/usr/bin/env rye-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2024 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import open3d as o3d
import rospkg
import rospy
import tf
import torch
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

from gedi import GeDi


class Registration(object):

    def __init__(self) -> None:
        pkg_path = rospkg.RosPack().get_path('gedi')

        config = {
            'dim': 32,  # descriptor output dimension
            'samples_per_batch': 500,  # batches to process the data on GPU
            'samples_per_patch_lrf': 4000,  # num. of point to process with LRF
            'samples_per_patch_out': 512,  # num. of points to sample for pointnet++
            'r_lrf': .5,  # LRF radius
            'fchkpt_gedi_net': pkg_path + '/data/chkpts/3dmatch/chkpt.tar'
        }  # path to checkpoint

        self.__voxel_size = .01
        self.__patches_per_pair = 5000

        # initialising class
        self.__gedi = GeDi(config=config)

        ref_pcd_path = rospy.get_param('reference_point_cloud', pkg_path + '/data/assets/shelf.pcd')
        # getting a pair of point clouds
        pcd = o3d.io.read_point_cloud(ref_pcd_path)
        inds = np.random.choice(np.asarray(pcd.points).shape[0], self.__patches_per_pair, replace=False)
        pts = torch.tensor(np.asarray(pcd.points)[inds]).float()
        pcd0 = pcd.voxel_down_sample(self.__voxel_size)
        pcd0_ = torch.tensor(np.asarray(pcd0.points)).float()
        pcd0_desc = self.__gedi.compute(pts=pts, pcd=pcd0_)
        self.__ref_pcd_dsdv = o3d.pipelines.registration.Feature()
        self.__ref_pcd_dsdv.data = pcd0_desc.T
        self.__ref_pcd = o3d.geometry.PointCloud()
        self.__ref_pcd.points = o3d.utility.Vector3dVector(pts)

        self.__pub = rospy.Publisher('/gedi/pose', PoseStamped, queue_size=10)

        rospy.Subscriber('/soar/head_camera/depth_registered/points', PointCloud2, self.__callback)

    def __callback(self, msg: PointCloud2) -> None:
        points = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.array(list(points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # pcd0 = self.__ref_pcd
        # pcd1 = pcd
        # pcd0.paint_uniform_color([1, 0.706, 0])
        # pcd1.paint_uniform_color([0, 0.651, 0.929])
        # o3d.visualization.draw_geometries([pcd0, pcd1])
        # estimating normals (only for visualisation)
        # pcd0.estimate_normals()
        # pcd1.estimate_normals()

        # randomly sampling some points from the point cloud
        inds1 = np.random.choice(np.asarray(pcd.points).shape[0], self.__patches_per_pair, replace=False)

        pts1 = torch.tensor(np.asarray(pcd.points)[inds1]).float()

        # applying voxelisation to the point cloud
        pcd1 = pcd.voxel_down_sample(self.__voxel_size)

        _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

        # computing descriptors
        pcd1_desc = self.__gedi.compute(pts=pts1, pcd=_pcd1)

        # preparing format for open3d ransac
        pcd1_dsdv = o3d.pipelines.registration.Feature()

        pcd1_dsdv.data = pcd1_desc.T

        _pcd1 = o3d.geometry.PointCloud()
        _pcd1.points = o3d.utility.Vector3dVector(pts1)

        # applying ransac
        est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.__ref_pcd,
            _pcd1,
            self.__ref_pcd_dsdv,
            pcd1_dsdv,
            mutual_filter=False,
            max_correspondence_distance=.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = est_result01.transformation[0, 3]
        pose.pose.position.y = est_result01.transformation[1, 3]
        pose.pose.position.z = est_result01.transformation[2, 3]
        q = tf.transformations.quaternion_from_matrix(est_result01.transformation)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.__pub.publish(pose)
        rospy.loginfo(est_result01)
        # applying estimated transformation
        # pcd0.transform(est_result01.transformation)
        # o3d.visualization.draw_geometries([pcd0, pcd1])


if __name__ == '__main__':
    rospy.init_node('gedi_registration')
    Registration()
    rospy.spin()
