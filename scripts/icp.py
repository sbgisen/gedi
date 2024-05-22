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
        device = o3d.core.Device('CPU:0')
        voxel_sizes = o3d.utility.DoubleVector([0.2, 0.1, 0.05])
        source = o3d.t.geometry.PointCloud().from_legacy(self.__ref_pcd)
        target = o3d.t.geometry.PointCloud().from_legacy(pcd)
        criteria_list = [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.001,
                                                                relative_rmse=0.001,
                                                                max_iteration=50),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                                                relative_rmse=0.0001,
                                                                max_iteration=50),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
            # o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 20),
            # o3d.t.pipelines.registration.ICPConvergenceCriteria(0.0000001, 0.0000001, 10)
        ]
        max_correspondence_distances = o3d.utility.DoubleVector([0.5, 0.3, 0.14])
        init_trans = o3d.core.Tensor(self.__trans_init, dtype=o3d.core.Dtype.Float32)

        def callback_after_iteration(loss_log_map): return print(
            'Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},'.
            format(loss_log_map['iteration_index'].item(), loss_log_map['scale_index'].item(), loss_log_map[
                'scale_iteration_index'].item(), loss_log_map['fitness'].item(), loss_log_map['inlier_rmse'].item()))
        reg_p2l = o3d.t.pipelines.registration.multi_scale_icp(
            source, target, voxel_sizes, criteria_list, max_correspondence_distances, init_trans,
            o3d.t.pipelines.registration.TransformationEstimationPointToPlane(), callback_after_iteration)
        # reg_p2l = o3d.pipelines.registration.registration_icp(
        #     self.__ref_pcd, pcd, threshold, self.__trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())

        rospy.loginfo(reg_p2l)
        rospy.loginfo('Transformation is:')
        rospy.loginfo(reg_p2l.transformation)
        self.draw_registration_result(source, target, reg_p2l.transformation)

    def draw_registration_result(self, source, target, transformation):
        source_temp = source.clone()
        target_temp = target.clone()
        source_temp2 = source.clone()
        source_temp2.transform(self.__trans_init)
        source_temp2.paint_uniform_color([0., 0.87, 0.3])
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp2.to_legacy(), source_temp.to_legacy(), target_temp.to_legacy()])


if __name__ == '__main__':
    rospy.init_node('gedi_registration')
    Registration()
    rospy.spin()
