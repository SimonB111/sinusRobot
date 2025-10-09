#!/usr/bin/env python

import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

class Robot:
    '''Class representing a robot'''

    def __init__(self, pose: PoseStamped = None) -> None:
        '''
        Creates a Robot object with position and orientation
        '''
        self.pose = pose
        
    def callback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from /measured_cp,
        updates Robot attributes. Logs info
        Returns: None
        '''
        self.pose = poseIn
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation
        rospy.loginfo(f"Position -> x: {pos.x}, y: {pos.y}, z: {pos.z}")
        rospy.loginfo(f"Orientation -> x: {ori.x}, y: {ori.y}, z: {ori.z}, w: {ori.w}")

    def createlistener(self) -> None:
        '''
        Creates a node that listens to /measured_cp
        Returns: None
        '''
        # init node with name listener
        rospy.init_node('listener', anonymous=True)
        # subscribe to measured_cp topic
        rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, 
                         self.callback)

        rospy.spin() # keep the node running 

 
if __name__ == '__main__':
    sinusRobot = Robot()
    sinusRobot.createlistener()