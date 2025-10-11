#!/usr/bin/env python

import rospy
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

class Robot:
    '''Class representing a robot'''

    def __init__(self, pose: PoseStamped = None) -> None:
        '''
        Creates a Robot object with position and orientation
        '''
        self.pose = pose
        self.plotter = BackgroundPlotter()
        self.actor = None
        
    def callback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from /measured_cp,
        updates Robot attributes. Logs info
        Returns: None
        '''
        self.pose = poseIn
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation
        rospy.loginfo(f"pos x: {pos.x:.3f}, y: {pos.y:.3f}, z: {pos.z:.3f}")
        rospy.loginfo(f"ori x: {ori.x:.3f}, y: {ori.y:.3f}, z: {ori.z:.3f}, w: {ori.w:.3f}")

    def createListener(self) -> None:
        '''
        Creates a node that listens to /measured_cp
        Returns: None
        '''
        # init node with name listener
        rospy.init_node('listener', anonymous=True)
        # subscribe to measured_cp topic
        rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, self.callback)

        rate = rospy.Rate(30)  # 30 Hz update rate
        while not rospy.is_shutdown():
            # update visuals
            if self.pose is not None:
                self.drawEndEffector()
                self.plotter.update()
            self.plotter.app.processEvents()
            rate.sleep()

    def drawEndEffector(self) -> None:
        '''
        Creates a basic 3D visualization of the end effector's
        position and orientation
        Returns: None
        '''
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation

        if self.actor is None:
            # create mesh once
            arrowX = pv.Arrow(start=(pos.x, pos.y, pos.z), 
                              direction=(1, 0, 0), scale=2)
            arrowZ = pv.Arrow(start=(pos.x, pos.y, pos.z), 
                              direction=(0, 0, 1), scale=1)
            self.effectorMesh = pv.merge([arrowX, arrowZ]) # combine meshes
            self.meshSave = self.effectorMesh.copy()
            self.actor = self.plotter.add_mesh(self.effectorMesh, color='blue')
        else :
            # convert quaternion to rot matrix
            rot = Rot.from_quat((ori.x, ori.y, ori.z, ori.w))
            rot_matrix = rot.as_matrix()

            # apply rot matrix to points in mesh
            points = self.meshSave.points.copy()
            rotatedPoints = points.dot(rot_matrix.T)
            self.effectorMesh.points = rotatedPoints + np.array([pos.x, pos.y, pos.z])

        self.plotter.show_axes()
        self.plotter.update() # update the display
 

if __name__ == '__main__':
    sinusRobot = Robot()
    sinusRobot.createListener()