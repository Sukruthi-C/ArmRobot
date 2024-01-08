"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from cv2 import contourArea
import kinematics
from scipy import ndimage
class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.teach_repeat_points=[]
        self.gripper_state_list=[]
        self.moving_interval=[]
        self.detect_location=[]
        self.detect_color=[]
        self.detect_theta=[]
        self.detect_area=[]
        # self.queue=list(({'color': 'red', 'num': 1},{'color': 'orange', 'num': 2},{'color': 'yellow', 'num': 3},{'color': 'green', 'num': 4},{'color': 'blue', 'num': 5},{'color': 'violet', 'num': 6}))
        self.queue=['red','orange','yellow','green','blue','violet']
        self.last_time=time.time()

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()
        if self.next_state == "picksort":
            self.picksort()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "record":
            self.record()
        if self.next_state == "stack":
            self.stack()
        if self.next_state == "line":
            self.line()
        if self.next_state == "colorstack":
            self.colorstack()
        if self.next_state == "stackhigh":
            self.stackhigh()



    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        
        self.status_message = "State: Execute - Executing motion plan"
        self.rxarm.set_moving_time(2)
        self.rxarm.set_accel_time(0.5)
        for i in range(len(self.teach_repeat_points)):
            self.rxarm.set_positions(self.teach_repeat_points[i])
            if(self.gripper_state_list[i]>=0.0375):
                self.rxarm.gripper.release()
            else:
                self.rxarm.gripper.grasp()
            time.sleep(1.1)
        self.next_state = "idle"


    def recover_homogenous_transform_pnp_ransac(self,image_points, world_points, K):
        '''
        Use SolvePnP to find the rigidbody transform representing the camera pose in
        world coordinates (not working)
        '''
        distCoeffs = self.camera.distortion_matrix
        [_, R_exp, t, _] = cv2.solvePnPRansac(world_points, image_points, K,
                                            distCoeffs)
        R, _ = cv2.Rodrigues(R_exp)
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

    def recover_homogeneous_transform_svd(self,m, d):
        ''' 
        finds the rigid body transform that maps m to d: 
        d == np.dot(m,R) + T
        http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
        '''
        # calculate the centroid for each set of points
        d_bar = np.sum(d, axis=0) / np.shape(d)[0]
        m_bar = np.sum(m, axis=0) / np.shape(m)[0]

        # we are using row vectors, so tanspose the first one
        # H should be 3x3, if it is not, we've done this wrong
        H = np.dot(np.transpose(d - d_bar), m - m_bar)
        [U, S, V] = np.linalg.svd(H)

        R = np.matmul(V, np.transpose(U))
        # if det(R) is -1, we've made a reflection, not a rotation
        # fix it by negating the 3rd column of V
        if np.linalg.det(R) < 0:
            V = [1, 1, -1] * V
            R = np.matmul(V, np.transpose(U))
        T = d_bar - np.dot(m_bar, R)
        return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))



    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.camera.loadCameraCalibration("/home/student_pm/armlabPro/calibration.yaml")

        points_uv=self.camera.tag_uv_cam
        points_uvd=self.camera.tag_uvd_cam
        depths_camera=np.transpose(self.camera.tag_d_cam)
        points_ones=np.ones((4,1))
        points_world=self.camera.tag_locations.astype(np.float32)
        # print(self.camera.tag_uvd_cam)
        points_camera=np.transpose(depths_camera*np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.transpose(np.column_stack((points_uv,points_ones)))))
        # print(points_camera)

        A_pnp = self.recover_homogenous_transform_pnp_ransac(points_uv.astype(np.float32), points_world, self.camera.intrinsic_matrix)
        A_svd = self.recover_homogeneous_transform_svd(points_world, points_camera)
        # print(self.camera.tag_uv_cam.shape,self.camera.tag_perspective_loc.shape)
        self.camera.perspectiveMat = cv2.getPerspectiveTransform(self.camera.tag_uv_cam, self.camera.tag_perspective_loc)
        # self.camera.perspectiveMat = cv2.getAffineTransform(self.camera.tag_uv_cam[0:3,:], self.camera.tag_perspective_loc[0:3,:])
        print(self.camera.perspectiveMat)
        # points_transformed_pnp = np.dot(np.linalg.inv(A_pnp), np.transpose(np.column_stack((points_camera, points_ones))))
        # print(A_pnp,points_transformed_pnp)
        self.camera.extrinsic_matrix=A_svd
        self.camera.cameraCalibrated=True
        print(A_svd)
        print(points_uv)


        self.next_state = "idle"
        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration, RES is shown in terminal"

    """ TODO """

    
    def detect(self,L,R):
        """!
        @brief      Detect the blocks
        """
        self.status_message = "detect"
        self.detect_color=[]
        self.detect_location=[]
        self.detect_theta=[]
        self.detect_area=[]
        if np.sum(self.camera.perspectiveMat)==3:
            depthImg = self.camera.DepthFrameHSV[:,:,0].copy()
            detectRes = self.camera.VideoFrame.copy()
        else:
            depthImg = cv2.warpPerspective(self.camera.DepthFrameHSV[:,:,0], self.camera.perspectiveMat, (self.camera.DepthFrameHSV[:,:,0].shape[1], self.camera.DepthFrameHSV[:,:,0].shape[0]))
            detectRes = cv2.warpPerspective(self.camera.VideoFrame, self.camera.perspectiveMat, (self.camera.VideoFrame.shape[1], self.camera.VideoFrame.shape[0]))
        mask = np.zeros_like(depthImg)
        cv2.rectangle(mask, (120,45),(1150,700), 255, cv2.FILLED)
        cv2.rectangle(mask, (550,450),(710,700), 0, cv2.FILLED)
        mask2 = np.zeros_like(depthImg)
        cv2.rectangle(mask2, (L,70),(R,650), 255, cv2.FILLED)
        cv2.rectangle(mask2, (540,360),(740,700), 0, cv2.FILLED)
        depthPro = cv2.bitwise_and(depthImg,mask)
        depthPro = cv2.medianBlur(depthPro,5)
        depthPro = cv2.Canny(depthPro,20,35)
        depthPro = cv2.bitwise_and(depthPro,mask2)
        # depthPro = self.camera.max_Filter(depthPro,5)
        depthPro = ndimage.maximum_filter(depthPro,size=5)
        depthPro[depthPro<100]=0
        contours, _ = cv2.findContours(depthPro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        i=1
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00']==0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            A = contourArea(contour)
            flag=False
            for i in range(len(self.detect_location)):
                if np.linalg.norm(np.array([cx,cy])-np.array(self.detect_location[i][0:2]))<40 or A<100:
                    flag=True
            if flag:
                continue
            self.detect_location.append([cx,cy,depthImg[cy,cx]*2+650])
            self.detect_color.append(self.camera.retrieve_area_color(detectRes, cx, cy, self.camera.colors))
            self.detect_theta.append(cv2.minAreaRect(contour)[2])
            self.detect_area.append(A)
            cv2.drawContours(detectRes, contour, -1, (0,255,255), thickness=1)
        print(self.detect_location,self.detect_color,self.detect_area)
        cv2.imwrite('/home/student_pm/armlabPro/src/detect.jpg',detectRes)
        self.next_state = 'idle'
        time.sleep(1)

    def record(self):
        self.status_message = "record position start"
        posTmp=self.rxarm.get_positions()
        self.teach_repeat_points.append(posTmp)
        self.gripper_state_list.append(self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index])
        # if pos of gripper is over 0.0375 it's open
        # print([posTmp,self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]])
        print(self.teach_repeat_points[0],self.gripper_state_list[0])
        self.next_state='idle'
        
    def picksort(self):
        self.status_message = "Pick&Sort Start"
        LN=0
        RN=0
        while(True):
            self.detect(400,850)
            print(self.detect_location)
            self.rxarm.initialize()
            time.sleep(2)
            if(len(self.detect_location)==0):
                break
            for i in range(len(self.detect_location)):
                pt_loc_old=np.array([self.detect_location[i][0],self.detect_location[i][1],1])
                if np.sum(self.camera.perspectiveMat)==3:
                    pt_loc=np.array([self.detect_location[i][0],self.detect_location[i][1],0])
                else:
                    pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
                # depth_img=cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.perspectiveMat, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
                # pt_loc[2] = depth_img[int(pt_loc[1])][int(pt_loc[0])]
                pt_loc[2] = np.min(self.camera.DepthFrameRaw[int(pt_loc[1])-7:int(pt_loc[1])+7,int(pt_loc[0])-7:int(pt_loc[0])+7])
                tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
                point_world=np.dot(np.linalg.inv(self.camera.extrinsic_matrix),np.array([tmp[0],tmp[1],tmp[2],1]).T)
                point_world[0]+=8
                point_world[1]+=20
                
                pose =np.array([point_world[0],point_world[1],point_world[2],np.arctan2(point_world[0],point_world[1]),0,np.pi])
                
                # #Move above the block
                # pose[2]=120
                # print(pose)
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)


                # #move down 
                # pose[2]=point_world[2]-15
                # if pose[2]<=15:
                #     pose[2]=15
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)

                # #Grab and lift
                # self.rxarm.gripper.grasp()
                # time.sleep(2)
                # pose[2]+=140
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)

                # #Sort and Move
                # # 0.022
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     RN+=1
                #     dest=np.array([300,RN*75,200,np.arctan2(300,RN*75+25),0,np.pi])
                # else:
                #     LN+=1
                #     dest=np.array([-300,LN*50,200,np.arctan2(-300,LN*50+25),0,np.pi])
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)


                # #Place and Finish
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     dest[2]=40
                # else:
                #     dest[2]=25
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)
                # self.rxarm.gripper.release()
                # time.sleep(2)

                # #Lift up for next
                # # dest[2]=150 #original
                # dest[2]=120#test
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)

                joint_angles=[]
                # Move above the block
                pose[2]=160
                joint_angles.append(kinematics.IK_Vert(pose))

                #move down 
                pose[2]=point_world[2]-10
                if pose[2]<=15:
                    pose[2]=15
                joint_angles.append(kinematics.IK_geometric(pose))

                #Grab and lift
                joint_angles.append([-10])
                pose[2]+=120
                if pose[2]>=160:
                    pose[2]=160
                joint_angles.append(kinematics.IK_Vert(pose))
                for item in joint_angles:
                    
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                joint_angles=[]
                #Sort and Move
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    RN+=1
                    dest=np.array([300,RN*50,160,np.arctan2(300,RN*50),0,np.pi])
                else:
                    LN+=1
                    dest=np.array([-300,LN*40,160,np.arctan2(-300,LN*40),0,np.pi])
                joint_angles.append(kinematics.IK_Vert(dest))


                #Place and Finish
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    dest[2]=40
                else:
                    dest[2]=25
                joint_angles.append(kinematics.IK_geometric(dest))
                joint_angles.append([-20])

                #Lift up for next
                dest[2]=150#test
                joint_angles.append(kinematics.IK_Vert(dest))
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                
                

            self.rxarm.sleep()
            time.sleep(2)
        
        

        self.status_message = "Pick&Sort End"
        self.next_state='idle'


    def stack(self):
        self.status_message = "Pick&Stack Start"
        LN=0
        RN=0
        while(True):
            self.detect(420,830)
            print(self.detect_location)
            self.rxarm.initialize()
            time.sleep(2)
            if(len(self.detect_location)==0):
                break
            for i in range(len(self.detect_location)):
                pt_loc_old=np.array([self.detect_location[i][0],self.detect_location[i][1],1])
                if np.sum(self.camera.perspectiveMat)==3:
                    pt_loc=np.array([self.detect_location[i][0],self.detect_location[i][1],0])
                else:
                    pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
                # depth_img=cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.perspectiveMat, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
                # pt_loc[2] = depth_img[int(pt_loc[1])][int(pt_loc[0])]
                pt_loc[2] = np.min(self.camera.DepthFrameRaw[int(pt_loc[1])-7:int(pt_loc[1])+7,int(pt_loc[0])-7:int(pt_loc[0])+7])
                tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
                point_world=np.dot(np.linalg.inv(self.camera.extrinsic_matrix),np.array([tmp[0],tmp[1],tmp[2],1]).T)
                point_world[0]+=8
                point_world[1]+=20
                
                pose =np.array([point_world[0],point_world[1],point_world[2],np.arctan2(point_world[0],point_world[1]),0,np.pi])
                
                #Move above the block
                # pose[2] = 200
                # joint_angles = kinematics.IK_geometric(pose)
                # print('A',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)


                # #move down 
                # pose[2]=point_world[2]-15
                # if pose[2]<=15:
                #     pose[2]=15
                # joint_angles = kinematics.IK_geometric(pose)
                # print('B',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)

                # #Grab and lift
                # self.rxarm.gripper.grasp()
                # time.sleep(2)
                # pose[2]+=200#changed from 140
                # joint_angles = kinematics.IK_geometric(pose)
                # print('C',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)

                # #Sort and Move
                # # 0.022
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     RN+=1
                #     dest=np.array([250,275,200,np.arctan2(250,275),0,np.pi])
                # else:
                #     LN+=1
                #     dest=np.array([-250,275,200,np.arctan2(-250,275),0,np.pi])
                # joint_angles = kinematics.IK_geometric(dest)
                # print('D',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)


                # #Place and Finish
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     dest[2]=40*RN
                # else:
                #     dest[2]=25*LN
                # joint_angles = kinematics.IK_geometric(dest)
                # print('E',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)
                # self.rxarm.gripper.release()
                # time.sleep(2)

                # #Lift up for next
                # dest[2]=160
                # joint_angles = kinematics.IK_geometric(dest)
                # print('F',joint_angles)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)
                joint_angles=[]
                # Move above the block
                pose[2]=160
                joint_angles.append(kinematics.IK_Vert(pose))

                #move down 
                pose[2]=point_world[2]-10
                if pose[2]<=15:
                    pose[2]=15
                joint_angles.append(kinematics.IK_geometric(pose))

                #Grab and lift
                # -10: grasp
                joint_angles.append([-10])
                pose[2]+=120
                if pose[2]>=160:
                    pose[2]=160
                joint_angles.append(kinematics.IK_Vert(pose))
                for item in joint_angles:
                    
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                joint_angles=[]
                #Sort and Move
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    RN+=1
                    dest=np.array([250,275,200,np.arctan2(250,275),0,np.pi])
                else:
                    LN+=1
                    dest=np.array([-250,275,200,np.arctan2(-250,275),0,np.pi])
                joint_angles.append(kinematics.IK_geometric(dest))


                #Place and Finish
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    dest[2]=40*RN
                else:
                    dest[2]=25*LN
                joint_angles.append(kinematics.IK_geometric(dest))
                joint_angles.append([-20])

                #Lift up for next
                dest[2]=150#test
                joint_angles.append(kinematics.IK_Vert(dest))
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)


            self.rxarm.sleep()
            time.sleep(2)
        
        

        self.status_message = "Pick&Stack End"
        self.next_state='idle'



    def line(self):
        self.status_message = "Line Start"
        while(True):
            self.detect(400,850)
            print(self.detect_location)
            self.rxarm.initialize()
            time.sleep(2)
            if(len(self.detect_location)==0):
                break
            for i in range(len(self.detect_location)):
                pt_loc_old=np.array([self.detect_location[i][0],self.detect_location[i][1],1])
                if np.sum(self.camera.perspectiveMat)==3:
                    pt_loc=np.array([self.detect_location[i][0],self.detect_location[i][1],0])
                else:
                    pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
                # depth_img=cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.perspectiveMat, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
                # pt_loc[2] = depth_img[int(pt_loc[1])][int(pt_loc[0])]
                pt_loc[2] = np.min(self.camera.DepthFrameRaw[int(pt_loc[1])-7:int(pt_loc[1])+7,int(pt_loc[0])-7:int(pt_loc[0])+7])
                tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
                point_world=np.dot(np.linalg.inv(self.camera.extrinsic_matrix),np.array([tmp[0],tmp[1],tmp[2],1]).T)
                point_world[0]+=8
                point_world[1]+=20
                
                pose =np.array([point_world[0],point_world[1],point_world[2],np.arctan2(point_world[0],point_world[1]),0,np.pi])
                
                # #Move above the block
                # pose[2] = 200
                # print(pose)
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)


                # #move down 
                # pose[2] = point_world[2]-15
                # if pose[2]<=15:
                #     pose[2]=15
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)

                # #Grab and lift
                # self.rxarm.gripper.grasp()
                # time.sleep(2)
                # pose[2]+=140
                # joint_angles = kinematics.IK_geometric(pose)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)

                # #Sort and Move
                # # 0.022
                # pos = 3
                # if(self.detect_color[i]=='red'):
                #     pos=1
                # elif (self.detect_color[i]=='orange'):
                #     pos=2
                # elif (self.detect_color[i]=='yellow'):
                #     pos=3
                # elif (self.detect_color[i]=='green'):
                #     pos=4
                # elif (self.detect_color[i]=='blue'):
                #     pos=5
                # elif (self.detect_color[i]=='violet'):
                #     pos=6
                # print(pos)
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     dest=np.array([300,pos*60-60,160,np.arctan2(300,pos*60-60),0,np.pi])
                # else:
                #     dest=np.array([-300,pos*40-40,160,np.arctan2(-300,pos*40-40),0,np.pi])
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)


                # #Place and Finish
                # if self.detect_area[i]>=1500 or self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                #     dest[2]=40
                # else:
                #     dest[2]=25
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(2)
                # self.rxarm.gripper.release()
                # time.sleep(2)

                # #Lift up for next
                # dest[2]=160
                # joint_angles = kinematics.IK_geometric(dest)
                # self.rxarm.set_positions(joint_angles)
                # time.sleep(3)
                joint_angles=[]
                # Move above the block
                pose[2]=200
                joint_angles.append(kinematics.IK_Vert(pose))

                #move down 
                pose[2]=point_world[2]-10
                if pose[2]<=15:
                    pose[2]=15
                joint_angles.append(kinematics.IK_geometric(pose))

                #Grab and lift
                joint_angles.append([-10])
                pose[2]+=120
                if pose[2]>=200:
                    pose[2]=200
                joint_angles.append(kinematics.IK_Vert(pose))
                for item in joint_angles:
                    
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                joint_angles=[]
                #Sort and Move
                pos = 3
                if(self.detect_color[i]=='red'):
                    pos=1
                elif (self.detect_color[i]=='orange'):
                    pos=2
                elif (self.detect_color[i]=='yellow'):
                    pos=3
                elif (self.detect_color[i]=='green'):
                    pos=4
                elif (self.detect_color[i]=='blue'):
                    pos=5
                elif (self.detect_color[i]=='violet'):
                    pos=6
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    dest=np.array([300,pos*50-20,160,np.arctan2(300,pos*50-20),0,np.pi])
                else:
                    dest=np.array([-300,pos*40-20,160,np.arctan2(-300,pos*40-20),0,np.pi])
                joint_angles.append(kinematics.IK_Vert(dest))


                #Place and Finish
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    dest[2]=40
                else:
                    dest[2]=25
                joint_angles.append(kinematics.IK_geometric(dest))
                joint_angles.append([-20])

                #Lift up for next
                dest[2]=150#test
                joint_angles.append(kinematics.IK_Vert(dest))
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)

            self.rxarm.sleep()
            time.sleep(2)
        
        

        self.status_message = "Line End"
        self.next_state='idle'


    def colorstack(self):
        self.status_message = "Color Stack Start"
        bigStacked=0
        smallStacked=0
        first= True
        while(True):
            self.detect(430,850)
            # print(self.detect_color,self.detect_location)
            self.rxarm.initialize()
            time.sleep(2)
            # self.rxarm.set_positions(np.array([-np.arctan2(point_world[0],point_world[1]),0,0,0,0]))
            for i in range(len(self.detect_location)-1):
                for j in range(i+1,len(self.detect_location)):
                    if self.queue.index(self.detect_color[i])>self.queue.index(self.detect_color[j]):
                        tmp = self.detect_location[j]
                        self.detect_location[j]=self.detect_location[i]
                        self.detect_location[i]=tmp

                        tmp = self.detect_area[j]
                        self.detect_area[j]=self.detect_area[i]
                        self.detect_area[i]=tmp

                        tmp = self.detect_color[j]
                        self.detect_color[j]=self.detect_color[i]
                        self.detect_color[i]=tmp
            
            print(self.detect_color,self.detect_location)
            if(len(self.detect_location)==0):
                break
            for i in range(len(self.detect_location)):
                pt_loc_old=np.array([self.detect_location[i][0],self.detect_location[i][1],1])
                if np.sum(self.camera.perspectiveMat)==3:
                    pt_loc=np.array([self.detect_location[i][0],self.detect_location[i][1],0])
                else:
                    pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
                pt_loc[2] = np.min(self.camera.DepthFrameRaw[int(pt_loc[1])-10:int(pt_loc[1])+10,int(pt_loc[0])-10:int(pt_loc[0])+10])
                tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
                point_world=np.dot(np.linalg.inv(self.camera.extrinsic_matrix),np.array([tmp[0],tmp[1],tmp[2],1]).T)
                point_world[0]+=8
                point_world[1]+=20
                
                pose =np.array([point_world[0],point_world[1],point_world[2],np.arctan2(point_world[0],point_world[1]),0,np.pi])
                
                joint_angles=[]
                # Move above the block
                pose[2]=260
                joint_angles.append(kinematics.IK_Vert(pose))
                # joint_angles.append(np.array([-np.arctan2(point_world[0],point_world[1]),0,0,0,0]))

                #move down 
                pose[2]=point_world[2]-12
                if pose[2]<=20:
                    pose[2]=20
                joint_angles.append(kinematics.IK_geometric(pose))

                #Grab and lift
                joint_angles.append([-10])
                pose[2]=260
                joint_angles.append(kinematics.IK_Vert(pose))
                # joint_angles.append(np.array([-np.arctan2(point_world[0],point_world[1]),0,0,0,0]))
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                
                joint_angles=[]
                #Sort and Move
                pos = 3
                flag = False
                if(self.detect_color[i]=='red'):
                    pos=1
                elif (self.detect_color[i]=='orange'):
                    pos=2
                elif (self.detect_color[i]=='yellow'):
                    pos=3
                elif (self.detect_color[i]=='green'):
                    pos=4
                elif (self.detect_color[i]=='blue'):
                    pos=5
                elif (self.detect_color[i]=='violet'):
                    pos=6
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    if pos == bigStacked+1:
                        dest=np.array([250,75,280,np.arctan2(250,75),0,np.pi])
                        bigStacked+=1
                        flag=True
                    else:
                        dest=np.array([350,pos*50-20,280,np.arctan2(350,pos*50-20),0,np.pi])
                else:
                    if pos == smallStacked+1:
                        dest=np.array([-250,75,280,np.arctan2(-250,75),0,np.pi])
                        smallStacked+=1
                        flag=True
                    else:
                        dest=np.array([-350,pos*40-20,280,np.arctan2(-350,pos*40-20),0,np.pi])
                # joint_angles.append(np.array([-np.arctan2(dest[0],dest[1]),0,0,0,0]))
                joint_angles.append(kinematics.IK_Vert(dest))


                #Place and Finish
                if self.rxarm.gripper.core.joint_states.position[self.rxarm.gripper.left_finger_index]>0.018:
                    if flag:
                        dest[2]=40*bigStacked-20
                    else:
                        dest[2]=40
                else:
                    if flag:
                        dest[2]=25*smallStacked-15
                    else:
                        dest[2]=25
                joint_angles.append(kinematics.IK_geometric(dest))
                joint_angles.append([-20])

                #Lift up for next
                dest[2]=260
                joint_angles.append(kinematics.IK_Vert(dest))
                # joint_angles.append(np.array([-np.arctan2(dest[0],dest[1]),0,0,0,0]))
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
    
            self.rxarm.sleep()
            time.sleep(2)
        
        self.rxarm.initialize()
        joint_angles=[]
        for i in range(smallStacked+1,7):
            startPose=np.array([-350,i*40-20,180,np.arctan2(-350,i*40-20),0,np.pi])
            joint_angles.append(kinematics.IK_geometric(startPose))
            startPose[2]=25
            joint_angles.append(kinematics.IK_geometric(startPose))
            joint_angles.append([-10])
            startPose[2]=180
            joint_angles.append(kinematics.IK_geometric(startPose))
            endPose=np.array([-250,75,180,np.arctan2(-250,75),0,np.pi])
            joint_angles.append(kinematics.IK_geometric(endPose))
            endPose[2]=25*i-15
            joint_angles.append(kinematics.IK_geometric(endPose))
            joint_angles.append([-20])
            endPose[2]=180
            joint_angles.append(kinematics.IK_geometric(endPose))
        for i in range(bigStacked+1,7):
            startPose=np.array([350,i*50-20,260,np.arctan2(350,i*50-20),0,np.pi])
            joint_angles.append(kinematics.IK_geometric(startPose))
            startPose[2]=40
            joint_angles.append(kinematics.IK_geometric(startPose))
            joint_angles.append([-10])
            startPose[2]=300
            joint_angles.append(kinematics.IK_Vert(startPose))
            endPose=np.array([250,75,300,np.arctan2(250,75),0,np.pi])
            joint_angles.append(kinematics.IK_geometric(endPose))
            endPose[2]=40*i-20
            joint_angles.append(kinematics.IK_geometric(endPose))
            joint_angles.append([-20])
            endPose[2]=300
            joint_angles.append(kinematics.IK_Vert(endPose))
        for item in joint_angles: 
            if item[0] == -10:
                self.rxarm.gripper.grasp()
            elif item[0]==-20:
                self.rxarm.gripper.release()
            else:
                self.rxarm.set_positions(item)
            time.sleep(2)
        self.rxarm.set_positions(np.array([-np.arctan2(250,75),0,0,0,0]))
        time.sleep(2)
        self.rxarm.sleep()
        self.status_message = "Color Stack End"
        self.next_state='idle'


    def stackhigh(self):
        self.status_message = "Pick&Stack Start"
        stacked=0
        while(True):
            self.detect(420,830)
            print(self.detect_location)
            self.rxarm.initialize()
            time.sleep(2)
            if(len(self.detect_location)==0):
                break
            for i in range(len(self.detect_location)):
                pt_loc_old=np.array([self.detect_location[i][0],self.detect_location[i][1],1])
                if np.sum(self.camera.perspectiveMat)==3:
                    pt_loc=np.array([self.detect_location[i][0],self.detect_location[i][1],0])
                else:
                    pt_loc=np.matmul(np.linalg.inv(self.camera.perspectiveMat),pt_loc_old)/np.matmul(np.linalg.inv(self.camera.perspectiveMat)[2],pt_loc_old)
                # depth_img=cv2.warpPerspective(self.camera.DepthFrameRaw, self.camera.perspectiveMat, (self.camera.DepthFrameRaw.shape[1], self.camera.DepthFrameRaw.shape[0]))
                # pt_loc[2] = depth_img[int(pt_loc[1])][int(pt_loc[0])]
                pt_loc[2] = np.min(self.camera.DepthFrameRaw[int(pt_loc[1])-7:int(pt_loc[1])+7,int(pt_loc[0])-7:int(pt_loc[0])+7])
                tmp=np.dot(np.linalg.inv(self.camera.intrinsic_matrix),np.array([pt_loc[0],pt_loc[1],1]).T)*pt_loc[2]
                point_world=np.dot(np.linalg.inv(self.camera.extrinsic_matrix),np.array([tmp[0],tmp[1],tmp[2],1]).T)
                point_world[0]+=8
                point_world[1]+=20
                
                pose =np.array([point_world[0],point_world[1],point_world[2],np.arctan2(point_world[0],point_world[1]),0,np.pi])
               
                joint_angles=[]
                # Move above the block
                pose[2]=160
                joint_angles.append(kinematics.IK_geometric(pose))

                #move down 
                pose[2]=point_world[2]-10
                if pose[2]<=25:
                    pose[2]=25
                joint_angles.append(kinematics.IK_geometric(pose))

                #Grab and lift
                # -10: grasp
                joint_angles.append([-10])
                pose[2] = stacked*40+50
                if pose[2]<=120:
                    pose[2]=120
                if stacked<=5:
                    joint_angles.append(kinematics.IK_Vert(pose))
                else:
                    pose[2]=200
                    joint_angles.append(kinematics.IK_Vert(pose))
                    joint_angles.append(np.array([0,0,0,0,0]))
                
                for item in joint_angles:
                    
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)
                stacked+=1
                joint_angles=[]
                dest = np.zeros((5,1))
                #Sort and Move
                if stacked<=5:
                    dest=np.array([250,75,pose[2],np.arctan2(250,75),0,np.pi])
                    joint_angles.append(kinematics.IK_Vert(dest))
                else:
                    dest=np.array([240,75,stacked*40+50,np.arctan2(240,75),0,np.pi])
                    joint_angles.append(kinematics.clamp(kinematics.IK_Horizon(dest)))


                #Place and Finish
                dest[2]=stacked*40
                if stacked<=5:
                    joint_angles.append(kinematics.IK_geometric(dest))
                else:
                    dest[2]-=20
                    joint_angles.append(kinematics.clamp(kinematics.IK_Horizon(dest)))
                joint_angles.append([-20])
                TMP=kinematics.clamp(kinematics.IK_Horizon(dest))
                # print("sssssss",TMP)
                #Lift up for next
                if stacked<=5:
                    dest[2]+=50#test
                    joint_angles.append(kinematics.IK_Vert(dest))
                else:
                    TMP[2]-=0.3
                    joint_angles.append(TMP)             
                    G=TMP.copy()
                    G[0]=0
                    joint_angles.append(G)
                print(joint_angles)
                for item in joint_angles:
                    if item[0] == -10:
                        self.rxarm.gripper.grasp()
                    elif item[0]==-20:
                        self.rxarm.gripper.release()
                    else:
                        self.rxarm.set_positions(item)
                    time.sleep(2)


            self.rxarm.sleep()
            time.sleep(2)
        
        

        self.status_message = "Stack High End"
        self.next_state='idle'

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
            time.sleep(5)