#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
from functools import partial

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
import kinematics
import resource.config_parse
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.click_status = 0
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress
        self.ui.videoDisplay.mousePressEvent = self.click_operation

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.gripper.release())
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(lambda: self.rxarm.gripper.grasp())
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser5.setText('Record waypoint')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record'))
        self.ui.btnUser6.setText('Detect')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'detect'))
        self.ui.btn_task1.setText('PickSort')
        self.ui.btn_task1.clicked.connect(partial(nxt_if_arm_init, 'picksort'))
        self.ui.btn_task2.setText('Stack')
        self.ui.btn_task2.clicked.connect(partial(nxt_if_arm_init, 'stack'))
        self.ui.btn_task3.setText('Line')
        self.ui.btn_task3.clicked.connect(partial(nxt_if_arm_init, 'line'))
        self.ui.btn_task4.setText('ColorStack')
        self.ui.btn_task4.clicked.connect(partial(nxt_if_arm_init, 'colorstack'))
        self.ui.btn_task5.setText('StackHigh')
        self.ui.btn_task5.clicked.connect(partial(nxt_if_arm_init, 'stackhigh'))
        # self.ui.btnUser5.setText('Record waypoint')
        # self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record'))

        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()


    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        DHM = kinematics.FK_dh(resource.config_parse.parse_dh_param_file("/home/student_pm/armlabPro/config/rx200_dh.csv"),self.rxarm.get_positions())
        [theta,psi,phi] = kinematics.get_euler_angles_from_T(DHM)
        self.ui.rdoutX.setText(str("%+.2f mm" % (DHM[0,3]*1000)))
        self.ui.rdoutY.setText(str("%+.2f mm" % (DHM[1,3]*1000)))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (DHM[2,3]*1000)))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (phi)))
        self.ui.rdoutTheta.setText(str("%+.2f" % (theta)))
        self.ui.rdoutPsi.setText(str("%+.2f" % (psi)))

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image,detect_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))
        if (self.ui.radioUsr3.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(detect_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            pt_loc_old=np.array([pt.x(),pt.y(),1])
            if np.sum(self.camera.perspectiveMat)==3:
                pt_loc=np.array([pt.x(),pt.y(),0])
            else:
                pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
            pt_loc[2] = self.camera.DepthFrameRaw[int(pt_loc[1])][int(pt_loc[0])]

            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (pt.x(),pt.y(), pt_loc[2]))
            color_info = self.camera.VideoFrame[int(pt_loc[1]),int(pt_loc[0])]
            self.ui.rdoutMouseRGB.setText("(%.0f,%.0f,%.0f)" %
                                             (color_info[0],color_info[1], color_info[2]))

            point_world=np.ones((1,3))
            if self.camera.cameraCalibrated:
                EXT=self.camera.extrinsic_matrix
               
            else:
                EXT=self.camera.extrinsic_manual
               
            tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
            point_world=np.dot(np.linalg.inv(EXT),np.array([tmp[0],tmp[1],tmp[2],1]).T)
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (point_world[0]+8,point_world[1]+20,point_world[2]))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.new_click = True
        # print(self.camera.last_click)

    def click_operation(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.click_status += 1

        point_world = np.zeros(3)
        if self.camera.DepthFrameRaw.any() != 0:
            pt_loc_old=np.array([pt.x(),pt.y(),1])
            if np.sum(self.camera.perspectiveMat)==3:
                pt_loc=np.array([pt.x(),pt.y(),0])
            else:
                pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
            pt_loc[2] = self.camera.DepthFrameRaw[int(pt_loc[1])][int(pt_loc[0])]


            point_world=np.ones((1,3))
            if self.camera.cameraCalibrated:
                EXT=self.camera.extrinsic_matrix
               
            else:
                EXT=self.camera.extrinsic_manual
               
            tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
            point_world=np.dot(np.linalg.inv(EXT),np.array([tmp[0],tmp[1],tmp[2],1]).T)
            
        pose =np.array([point_world[0]+8,point_world[1]+20,point_world[2],np.arctan2(point_world[0]+8,point_world[1]+20),0,np.pi])
        joint_angles = kinematics.IK_geometric(pose)
        # print(np.rad2deg(joint_angles),pose)
        pose[2]+=100
        joint_angles = kinematics.IK_geometric(pose)
        self.rxarm.set_positions(joint_angles)
        time.sleep(3)
        pose[2]-=110
        if pose[2]<=25:
            pose[2]=25
        print(pose)
        joint_angles = kinematics.IK_geometric(pose)
        self.rxarm.set_positions(joint_angles)
        time.sleep(3)
        if self.click_status%2==1:#grab      
            self.rxarm.gripper.grasp()
        else:#place
            self.rxarm.gripper.release()
        pose[2]+=100
        joint_angles = kinematics.IK_geometric(pose)
        self.rxarm.set_positions(joint_angles)
        
        # print(self.camera.last_click)

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'])
    app_window.show()

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
