#!/usr/bin/env python
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

import rospy
from geometry_msgs.msg import TransformStamped
from jsk_recognition_msgs.msg import ICPResult
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class ReferencePub:

    def __init__(self):
        self.pub = rospy.Publisher('reference_offsets', PointCloud2, queue_size=1)
        self.rate = rospy.Rate(10)
        self.ref_points = rospy.wait_for_message('reference_points', PointCloud2)
        self.default_transform = TransformStamped()
        self.default_transform.transform.translation.x = 4.05
        self.default_transform.transform.translation.y = 3.65
        self.default_transform.transform.rotation.z = 0.7512804
        self.default_transform.transform.rotation.w = 0.6599831
        offset_points = do_transform_cloud(self.ref_points, self.default_transform)
        self.pub.publish(offset_points)
        self.sub = rospy.Subscriber('icp_result', ICPResult, self.publish)

    def publish(self, msg: ICPResult) -> None:
        transform = TransformStamped()
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z
        transform.transform.rotation.x = msg.pose.orientation.x
        transform.transform.rotation.y = msg.pose.orientation.y
        transform.transform.rotation.z = msg.pose.orientation.z
        transform.transform.rotation.w = msg.pose.orientation.w
        offset_points = do_transform_cloud(self.ref_points, transform)
        self.pub.publish(offset_points)


if __name__ == '__main__':
    rospy.init_node('reference_pub')
    _ = ReferencePub()
    rospy.spin()
