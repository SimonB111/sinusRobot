#!/usr/bin/env python

import rospy
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QTimer
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
        self.endoPose = None
        self.plotter = BackgroundPlotter()
        self.actor = None
        
    def callback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from /REMS/Research/measured_cp,
        updates Robot attributes. Logs info
        Returns: None
        '''
        self.pose = poseIn
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation
        rospy.loginfo(f"pos x: {pos.x:.3f}, y: {pos.y:.3f}, z: {pos.z:.3f}")
        rospy.loginfo(f"ori x: {ori.x:.3f}, y: {ori.y:.3f}, z: {ori.z:.3f}, w: {ori.w:.3f}")

    def endoCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from NDI/Endoscope/measured_cp,
        updates Robot attributes. Logs info
        Returns: None
        '''
        self.endoPose = poseIn
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation
        rospy.loginfo(f"pos x: {pos.x:.3f}, y: {pos.y:.3f}, z: {pos.z:.3f}")
        rospy.loginfo(f"ori x: {ori.x:.3f}, y: {ori.y:.3f}, z: {ori.z:.3f}, w: {ori.w:.3f}")

    def update(self) -> None:
        '''
        Calls draw function if we have a valid pose
        '''
        if self.pose is not None:
            self.drawEndEffector()

    def runListener(self) -> None:
        '''
        Creates nodes that listen to needed topics,
        loops continuously
        Returns: None
        '''
        # subscribe to measured_cp topic (PoseStamped)
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, self.callback)

        # subscribe to NDI/Endoscope/measured_cp ()
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("NDI/Endoscope/measured_cp", PoseStamped, self.endoCallback)

        timer = QTimer()
        # schedule task without blocking UI
        timer.timeout.connect(self.update)
        timer.start(40) # 25 Hz (too fast refresh freezes sooner)
        self.plotter.app.exec_()

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
                              direction=(1, 0, 0), scale=1)
            arrowY = pv.Arrow(start=(pos.x, pos.y, pos.z), 
                              direction=(0, 1, 0), scale=1)
            arrowZ = pv.Arrow(start=(pos.x, pos.y, pos.z), 
                              direction=(0, 0, 1), scale=1)
            # combine meshes
            self.effectorMesh = pv.merge([arrowX, arrowY, arrowZ]) 
            self.meshSave = self.effectorMesh.copy()
            self.actor = self.plotter.add_mesh(self.effectorMesh, color='red')
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
    sinusRobot.runListener()