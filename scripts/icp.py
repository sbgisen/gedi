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

import copy

import numpy as np
import open3d as o3d
import rospkg
import rospy
import tf
import tf.transformations
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


class Registration(object):

    def __init__(self) -> None:
        pkg_path = rospkg.RosPack().get_path('gedi')

        ref_pcd_path = rospy.get_param('~reference_point_cloud', pkg_path + '/data/assets/shelf.pcd')
        init_transform = rospy.get_param('~transform', [0, 0, 0, 0, 0, 0, 1])
        # getting a pair of point clouds
        pcd = o3d.io.read_point_cloud(ref_pcd_path)
        mat = tf.transformations.quaternion_matrix(init_transform[3:])
        mat[:3, 3] = init_transform[:3]
        self.__trans_init = mat
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
        self.__ref_pcd = pcd
        rospy.Subscriber('pass_through/output', PointCloud2, self.__callback)
        rospy.loginfo('Ready to register point clouds.')

    def __callback(self, msg: PointCloud2) -> None:
        points = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.array(list(points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))

        threshold = 0.02
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.__ref_pcd, pcd, threshold, self.__trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        rospy.loginfo(reg_p2l)
        rospy.loginfo('Transformation is:')
        rospy.loginfo(reg_p2l.transformation)
        self.draw_registration_result(self.__ref_pcd, pcd, reg_p2l.transformation)

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp2 = copy.deepcopy(source)
        source_temp2.transform(self.__trans_init)
        source_temp2.paint_uniform_color([0, 1, 0])
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp2, source_temp, target_temp])


if __name__ == '__main__':
    rospy.init_node('gedi_registration')
    Registration()
    rospy.spin()
