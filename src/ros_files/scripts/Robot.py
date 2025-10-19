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
        rospy.Subscriber("NDI/Endoscope/measured_cp", PoseStamped, self.endoCallback)

        timer = QTimer()
        # schedule task without blocking UI
        timer.timeout.connect(self.update)
        timer.start(40) # 25 Hz (too fast refresh freezes sooner)
        self.plotter.app.exec_()

    def draw(self) -> None:
        '''
        Creates a basic 3D visualization of the end effector's
        position and orientation
        Returns: None
        '''
        pos = self.pose.pose.position
        ori = self.pose.pose.orientation

        ePos = self.endoPose.pose.position
        eOri = self.endoPose.pose.orientation

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
            self.arrowMeshSave = self.effectorMesh.copy()
            self.actor = self.plotter.add_mesh(self.effectorMesh, color='red')

            self.endoMesh = self.effectorMesh.copy()
            self.endoActor = self.plotter.add_mesh(self.endoMesh, color='green')
        else :
            # convert quaternion to rot matrix
            rot = Rot.from_quat((ori.x, ori.y, ori.z, ori.w))
            rot_matrix = rot.as_matrix()
            # apply rot matrix to points in mesh
            points = self.arrowMeshSave.points.copy()
            rotatedPoints = points.dot(rot_matrix.T)
            self.effectorMesh.points = rotatedPoints + np.array([pos.x, pos.y, pos.z])

            # convert quaternion to rot matrix
            rot = Rot.from_quat((eOri.x, eOri.y, eOri.z, eOri.w))
            rot_matrix = rot.as_matrix()
            # apply rot matrix to points in mesh
            points = self.arrowMeshSave.points.copy()
            rotatedPoints = points.dot(rot_matrix.T)
            self.endoMesh.points = rotatedPoints + np.array([ePos.x, ePos.y, ePos.z])

        self.plotter.show_axes()
        self.plotter.update() # update the display
  

if __name__ == '__main__':
    sinusRobot = Robot()
    sinusRobot.runListeners()