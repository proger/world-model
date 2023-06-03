"""
This is a ROS node. This code needs to be dropped to https://github.com/EliaCereda/thymio_example to run.

See also https://github.com/jeguzzi/mighty-thymio for more context
"""
import gzip
import sys
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import TwistWithCovariance, PoseWithCovariance
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Range, Imu, LaserScan, Image
from nav_msgs.msg import Odometry
import tf_transformations

import numpy as np
from PIL import Image as PILImage


class RandomPolicy:
    def __init__(self, logger):
        self.rotate = None
        self.steps = 0
        self._logger = logger

    def get_logger(self):
        return self._logger

    def forward(self, image: np.ndarray, proximity: np.ndarray, right_sensor: float) -> Twist:
        cmd_vel = Twist()

        proximity_activations = np.abs(proximity) < 0.1
        any_proximity_activations = np.any(proximity_activations)
        right_obstacle = np.abs(right_sensor) < 0.05

        def flip():
            return np.random.rand(1)[0]

        self.steps += 1
 
        if not self.rotate:
            cmd_vel.linear.x = 5.0

            flip_ = flip()
            if any_proximity_activations or flip_ < 0.0001 or self.steps > 120:
                cmd_vel.linear.x = 0.0
                #self.get_logger().info(f"{proximity_activations} {flip_}. rotating", throttle_duration_sec=0.5)
                self.rotate = 1. if flip() < 0.5 else -1.
                if right_obstacle and self.rotate > 0:
                    self.rotate = -1.
                self.steps = 0
        else:
            cmd_vel.angular.z = self.rotate * 2.5
            if not any_proximity_activations and flip() < 0.1:
                cmd_vel.angular.z = 0.0
                self.rotate = None
                #self.get_logger().info(f"seeking adventure", throttle_duration_sec=0.5)
                self.steps = 0

        return cmd_vel


class Recorder:
    def __init__(self):
        self.pose_file = gzip.open('pose.bin.gz', mode='wb', compresslevel=3)
        self.twist_file = gzip.open('twist.bin.gz', mode='wb', compresslevel=3)
        self.image_file = gzip.open('image.bin.gz', mode='wb', compresslevel=3)
        self.proximity_file = gzip.open('proximity.bin.gz', mode='wb', compresslevel=3)
        self.counter = 0

    def forward(self,
                *,
                image: np.ndarray,
                proximity: np.ndarray,
                pose3: Optional[Pose],
                twist_action: Twist):
        if pose3:
            quaternion = (
                pose3.orientation.x,
                pose3.orientation.y,
                pose3.orientation.z,
                pose3.orientation.w
            )

            roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)

            pose_vector = np.array([
                pose3.position.x,  # x position
                pose3.position.y,  # y position
                yaw                # theta orientation
            ])
        else:
            pose_vector = np.array([-99999., -99999., -99999.])
        pose_vector.tofile(self.pose_file) # this was a bug

        twist_vector = np.array([twist_action.linear.x, twist_action.angular.z])
        self.twist_file.write(twist_vector.tobytes())

        self.image_file.write(image.tobytes())
        self.proximity_file.write(proximity.tobytes())

        self.counter += 1
        return self.counter


class RNNNode(Node):
    def __init__(self):
        super().__init__('rnn_node')

        self.ranges = {}
        self.proximity_subscribers = [
            self.create_subscription(Range, k, self.make_on_proximity(k), 1)
            for k in ['proximity/left', 'proximity/right',
                      'proximity/center', 'proximity/center_left', 'proximity/center_right',
                      'proximity/rear_left', 'proximity/rear_right',
                      'ground/left', 'ground/right']
        ]

        self.odometry = None
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.on_odometry, 1)

        self.image_subscriber = self.create_subscription(Image, 'camera', self.on_camera, 1)

        self.policy = RandomPolicy(self.get_logger())
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.recorder = Recorder()

    def on_camera(self, msg: Image):
        stamp = msg.header.stamp
        rows, cols = msg.height, msg.width
        assert msg.encoding == 'rgb8'
        assert msg.step == cols * 3

        image = np.array(msg.data, dtype=np.uint8).reshape(rows, cols, 3)
        image = np.array(PILImage.fromarray(image).resize((64, 64), resample=PILImage.NEAREST))

        proximity = np.array([
            self.ranges.get(k, -1.) for k in ['proximity/left', 'proximity/right',
                                              'proximity/center', 'proximity/center_left', 'proximity/center_right']
        ])

        odometry: Optional[Odometry] = self.odometry

        twist_action: Twist = self.policy.forward(image, proximity, self.ranges.get('proximity/right', -1.))
        self.twist_publisher.publish(twist_action)

        records = self.recorder.forward(image=image,
                                        proximity=proximity,
                                        pose3=odometry.pose.pose if odometry else None,
                                        twist_action=twist_action)

        if records % 1000 == 0:
            self.get_logger().info(f'have {records} records')

        if records >= 100000000:
            del self.recorder # close gzip objects
            import gc; gc.collect()
            self.get_logger().info('stopping')
            raise KeyboardInterrupt()

    def on_odometry(self, msg: Odometry):
        self.odometry = msg

    def make_on_proximity(self, k):
        def callback(msg):
            self.ranges[k] = msg.range
        return callback

    def start(self):
        pass
    
    def stop(self):
        # Set all velocities to zero
        self.twist_publisher.publish(Twist())


def main():
    # Initialize the ROS client library
    rclpy.init(args=sys.argv)
    
    # Create an instance of your node class
    node = RNNNode()
    node.start()
    
    # Keep processings events until someone manually shuts down the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Ensure the Thymio is stopped before exiting
    node.stop()


if __name__ == '__main__':
    main()
