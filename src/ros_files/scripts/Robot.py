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

    def __init__(self) -> None:
        '''
        Creates a Robot object with position and orientation
        '''
        self.pose = None
        self.endoPose = None
        self.anatPose = None
        self.plotter = BackgroundPlotter()
        self.actor = None
        self.endoActor = None
        
    def callback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from /REMS/Research/measured_cp,
        updates Robot attributes.
        Returns: None
        '''
        self.pose = poseIn

    def endoCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from NDI/Endoscope/measured_cp,
        updates Robot attributes.
        Returns: None
        '''
        self.endoPose = poseIn
    
    def anatCallback(self, poseIn: PoseStamped) -> None:
        '''
        Invoked when receiving data from NDI/Endoscope/measured_cp,
        updates Robot attributes.
        Returns: None
        '''
        self.anatPose = poseIn
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation
        rospy.loginfo(f"pos x: {pos.x:.3f}, y: {pos.y:.3f}, z: {pos.z:.3f}")
        rospy.loginfo(f"ori x: {ori.x:.3f}, y: {ori.y:.3f}, z: {ori.z:.3f}, w: {ori.w:.3f}")

    def update(self) -> None:
        '''
        Calls draw function if we have a valid pose
        '''
        if self.pose is not None and self.endoPose is not None and self.anatPose is not None:
            self.draw()

    def runListeners(self) -> None:
        '''
        Creates nodes that listen to needed topics,
        loops continuously
        Returns: None
        '''
        # initialize node and subscribe to appropriate topics
        rospy.init_node('listeners', anonymous=True)
        rospy.Subscriber("/REMS/Research/measured_cp", PoseStamped, self.callback)
        rospy.Subscriber("/NDI/Endoscope/measured_cp", PoseStamped, self.endoCallback)
        rospy.Subscriber("/NDI/Anatomy/measured_cp", PoseStamped, self.anatCallback)

        timer = QTimer()
        # schedule task without blocking UI
        timer.timeout.connect(self.update)
        timer.start(65) # (too fast refresh freezes sooner)
        self.plotter.app.exec_()

    def transformAxes(self, poseIn: PoseStamped) -> None:
        '''
        Creates a rotated and translated version of the standard 
        axes to visualize the orientation
        Parameters:
            poseIn: PoseStamped, the pose we want to translate to
        Returns: 
            output: pyvista_ndarray, the transformed points
        '''
        pos = poseIn.pose.position
        ori = poseIn.pose.orientation
        
        # convert quaternion to rot matrix
        rot = Rot.from_quat((ori.x, ori.y, ori.z, ori.w))
        rot_matrix = rot.as_matrix()
        # apply rot matrix to points in mesh
        points = self.arrowMeshSave.points.copy()
        rotatedPoints = points.dot(rot_matrix.T)
        return rotatedPoints + np.array([pos.x, pos.y, pos.z])

    def draw(self) -> None:
        '''
        Creates a basic 3D visualization of the position and orientation of
        the topics that are subscribed to
        Returns: None
        '''
        if self.actor is None:
            # create mesh once
            cubeX = pv.Cube(center=(.5,0,0), x_length=1, y_length=0.1, z_length=0.1)
            cubeY = pv.Cube(center=(0,.5,0), x_length=0.1, y_length=1, z_length=0.1)
            cubeZ = pv.Cube(center=(0,0,.5), x_length=0.1, y_length=0.1, z_length=1)
            # combine meshes
            self.effectorMesh = pv.merge([cubeX, cubeY, cubeZ])

            # copy meshes and assign to actor for each pose we visualize
            self.arrowMeshSave = self.effectorMesh.copy()
            self.actor = self.plotter.add_mesh(self.effectorMesh, color='red')

            self.endoMesh = self.effectorMesh.copy()
            self.endoActor = self.plotter.add_mesh(self.endoMesh, color='green')
            
            self.anatMesh = self.effectorMesh.copy()
            self.anatActor = self.plotter.add_mesh(self.anatMesh, color='blue')

            self.plotter.show_axes() # only need to call once
        else :
            # transform (rotate and translate)
            self.effectorMesh.points = self.transformAxes(self.pose)
            self.endoMesh.points = self.transformAxes(self.endoPose)
            self.anatMesh.points = self.transformAxes(self.anatPose)

        self.plotter.update() # update the display
  

if __name__ == '__main__':
    sinusRobot = Robot()
    sinusRobot.runListeners()