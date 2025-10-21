#!/usr/bin/env python

import rospy
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtCore import QTimer
import cv2
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

        self.nTicks = 0

        self.handEyeIsCalibrated = False
        self.sampleCount = 0
        self.maxSamples = 16
        # allocate arrays with appropriate shape
        self.tHand = np.zeros((self.maxSamples, 3)) 
        self.tEye = np.zeros((self.maxSamples, 3))
        self.rHand = np.zeros((self.maxSamples, 3, 3))
        self.rEye = np.zeros((self.maxSamples, 3, 3))
        # to be filled by calibrateHandEye,
        # will be homogenous transform from cam to gripper
        self.T_cam2Gripper = np.eye(4) 

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
        if not self.handEyeIsCalibrated and self.nTicks % 2 == 0:
            # run at half update rate to prevent large mismatch between REMS/NDI
            self.collectHandEye(self.pose, self.endoPose)
        elif not self.handEyeIsCalibrated:
            self.nTicks += 1 # increment ticks while in calibration phase
        elif self.pose is not None:
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
        timer.start(65) # ~15Hz (too fast refresh freezes sooner)
        self.plotter.app.exec_()

    def collectHandEye(self, handPose: PoseStamped, eyePose: PoseStamped):
        '''
        Collects arrays full of translation vectors and
        rotation matrices for both the hand and the eye.
        Calls calibrateHandEye() when full and forms
        self.T_cam2Gripper using the outputs
        Parameters:
            handPose: PoseStamped, pose in hand coords
            eyePose: PoseStamped, corresponding pose in eye coords
        '''
        if self.sampleCount < self.maxSamples:
            hPos = handPose.pose.position
            hOri = handPose.pose.orientation
            ePos = eyePose.pose.position
            eOri = eyePose.pose.orientation

            # fill current row with position vector
            self.tHand[self.sampleCount, :] = [hPos.x, hPos.y, hPos.z]
            self.tEye[self.sampleCount, :] = [ePos.x, ePos.y, ePos.z]

            # turn quat to rot matrix and assign for rHand and rEye
            hRot = Rot.from_quat((hOri.x, hOri.y, hOri.z, hOri.w))
            self.rHand[self.sampleCount] = hRot.as_matrix()

            eRot = Rot.from_quat((eOri.x, eOri.y, eOri.z, eOri.w))
            self.rHand[self.sampleCount] = eRot.as_matrix()

            self.sampleCount += 1 # move to next position
        else: # run when we have all samples
            rCam2Gripper, tCam2Gripper = cv2.calibrateHandEye(
                self.rHand, self.tHand, self.rEye, self.tEye)
            
            self.T_cam2Gripper[:3, :3] = rCam2Gripper # rotation part
            self.T_cam2Gripper[:3, 3] = tCam2Gripper.flatten() # translation part

            self.handEyeIsCalibrated = True

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

    def applyHomogenousTransform(self, points: pv.pyvista_ndarray, 
                                 transform) -> pv.pyvista_ndarray:
        '''
        Applies a given 4x4 homogeneous transformation matrix to a set of points.
        Parameters:
            points: pv.pyvista_ndarray, points to be transformed
            transform: the 4x4 homogeneous transformation matrix
        Returns: 
            output: pv.pyvista_ndarray, the transformed points
        '''
        # convert to homogenous points (need correct shape so we can do matrix mult)
        N = points.shape[0]
        homogeneousPoints = np.hstack((points, np.ones((N, 1)))) # now shape (N, 4)

        # (transpose for matrix multiplication), output shape (N,4)
        transformed_homogeneous = (transform @ homogeneousPoints.T).T  

        # convert back to 3D by removing the homogeneous coordinate
        transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3, np.newaxis]
        return transformed_points

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

            # these have extra steps since we need to transform these into
            # the same frame as end effector from REMS/research/measured_cp
            self.endoMesh.points = self.transformAxes(self.endoPose)
            self.endoMesh.points = self.applyHomogenousTransform(
                self.endoMesh.points, self.T_cam2Gripper)
            
            self.anatMesh.points = self.transformAxes(self.anatPose)
            self.anatMesh.points = self.applyHomogenousTransform(
                self.anatMesh.points, self.T_cam2Gripper)

        self.plotter.update() # update the display
  

if __name__ == '__main__':
    sinusRobot = Robot()
    sinusRobot.runListeners()