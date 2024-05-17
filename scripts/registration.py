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
import typing

import numpy as np
import open3d as o3d
import rospkg
import rospy
import teaserpp_python
import tf
import torch
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sklearn.neighbors import KDTree

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

        self.__voxel_size = rospy.get_param('~voxel_size', .02)
        self.__patches_per_pair = rospy.get_param('~patches_per_pair', 5000)

        # initialising class
        self.__gedi = GeDi(config=config)

        ref_pcd_path = rospy.get_param('~reference_point_cloud', pkg_path + '/data/assets/shelf.pcd')
        # getting a pair of point clouds
        pcd = o3d.io.read_point_cloud(ref_pcd_path)
        inds = np.random.choice(np.asarray(pcd.points).shape[0], self.__patches_per_pair, replace=False)
        pts = torch.tensor(np.asarray(pcd.points)[inds]).float()
        self.__pcd0 = pcd.voxel_down_sample(self.__voxel_size)
        # mean = np.mean(np.asarray(self.__pcd0.points), axis=0)
        # random_floor_x = np.random.uniform(mean[0] - 3, mean[0] + 3, int(len(self.__pcd0.points)*1))
        # random_floor_y = np.random.uniform(mean[1] - 3, mean[1] + 3, int(len(self.__pcd0.points) * 1))
        # random_floor_z = np.array([0] * int(len(self.__pcd0.points) * 1))
        # self.__pcd0.points.extend(
        #     o3d.utility.Vector3dVector(np.array([random_floor_x, random_floor_y, random_floor_z]).T))

        pcd0_ = torch.tensor(np.asarray(self.__pcd0.points)).float()
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
        # mean = np.mean(np.asarray(pcd1.points), axis=0)
        # random_floor_x = np.random.uniform(mean[0] - 3, mean[0] + 3, int(len(pcd1.points) * 1))
        # random_floor_y = np.random.uniform(mean[1] - 3, mean[1] + 3, int(len(pcd1.points) * 1))
        # random_floor_z = np.array([0] * int(len(pcd1.points) * 1))
        # pcd1.points.extend(o3d.utility.Vector3dVector(np.array([random_floor_x, random_floor_y, random_floor_z]).T))

        _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

        # computing descriptors
        start = rospy.Time.now()
        pcd1_desc = self.__gedi.compute(pts=pts1, pcd=_pcd1)
        rospy.loginfo(f'Gedi time: {(rospy.Time.now() - start).to_sec()}')

        # preparing format for open3d ransac
        pcd1_dsdv = o3d.pipelines.registration.Feature()

        pcd1_dsdv.data = pcd1_desc.T

        _pcd1 = o3d.geometry.PointCloud()
        _pcd1.points = o3d.utility.Vector3dVector(pts1)
        method = 'teaser'
        if method == 'ransac':

            # applying ransac
            start = rospy.Time.now()
            est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                self.__ref_pcd,
                _pcd1,
                self.__ref_pcd_dsdv,
                pcd1_dsdv,
                mutual_filter=False,
                max_correspondence_distance=.02,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=10,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))
            rospy.loginfo(f'Ransac time: {(rospy.Time.now() - start).to_sec()}')

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
        else:
            ref_matched_key, test_matched_key = self.find_mutually_nn_keypoints(self.__ref_pcd, _pcd1,
                                                                                self.__ref_pcd_dsdv, pcd1_dsdv)
            ref_matched_key = np.squeeze(ref_matched_key)
            test_matched_key = np.squeeze(test_matched_key)
            est_mat, _ = self.execute_teaser_global_registration(ref_matched_key, test_matched_key)
            a_pcd_t = copy.deepcopy(self.__pcd0).transform(est_mat)
            o3d.visualization.draw_geometries([a_pcd_t, pcd1])

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = est_mat[0, 3]
            pose.pose.position.y = est_mat[1, 3]
            pose.pose.position.z = est_mat[2, 3]
            q = tf.transformations.quaternion_from_matrix(est_mat)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            self.__pub.publish(pose)
            rospy.loginfo(est_mat)

    def find_mutually_nn_keypoints(self, ref_key: o3d.geometry.PointCloud, test_key: o3d.geometry.PointCloud,
                                   ref: o3d.pipelines.registration.Feature,
                                   test: o3d.pipelines.registration.Feature) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Use kdtree to find mutually closest keypoints.

        ref_key: reference keypoints (source)
        test_key: test keypoints (target)
        ref: reference feature (source feature)
        test: test feature (target feature)
        """
        ref_features = ref.data.T
        test_features = test.data.T
        ref_keypoints = np.asarray(ref_key.points)
        test_keypoints = np.asarray(test_key.points)
        n_samples = test_features.shape[0]

        ref_tree = KDTree(ref_features)
        test_tree = KDTree(test.data.T)
        test_nn_idx = ref_tree.query(test_features, return_distance=False)
        ref_nn_idx = test_tree.query(ref_features, return_distance=False)

        # find mutually closest points
        ref_match_idx = np.nonzero(np.arange(n_samples) == np.squeeze(test_nn_idx[ref_nn_idx]))[0]
        ref_matched_keypoints = ref_keypoints[ref_match_idx]
        test_matched_keypoints = test_keypoints[ref_nn_idx[ref_match_idx]]

        return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)

    def execute_teaser_global_registration(self, source: np.ndarray,
                                           target: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Use TEASER++ to perform global registration."""
        # Prepare TEASER++ Solver
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = self.__voxel_size
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

        # Solve with TEASER++
        teaserpp_solver.solve(source, target)
        est_solution = teaserpp_solver.getSolution()
        est_mat = self.compose_mat4_from_teaserpp_solution(est_solution)
        max_clique = teaserpp_solver.getTranslationInliersMap()
        return est_mat, max_clique

    def compose_mat4_from_teaserpp_solution(self, solution: teaserpp_python.RegistrationSolution) -> np.ndarray:
        """Compose a 4-by-4 matrix from teaserpp solution."""
        s = solution.scale
        rot_r = solution.rotation
        t = solution.translation
        t_mat = np.eye(4)
        t_mat[0:3, 3] = t
        r_mat = np.eye(4)
        r_mat[0:3, 0:3] = rot_r
        m_mat = t_mat.dot(r_mat)

        if s == 1:
            m_mat = t_mat.dot(r_mat)
        else:
            s_mat = np.eye(4)
            s_mat[0:3, 0:3] = np.diag([s, s, s])
            m_mat = t_mat.dot(r_mat).dot(s_mat)

        return m_mat


if __name__ == '__main__':
    rospy.init_node('gedi_registration')
    Registration()
    rospy.spin()
