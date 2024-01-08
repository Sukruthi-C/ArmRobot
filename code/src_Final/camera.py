#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import yaml
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from cv2 import contourArea
from PIL import ImageFilter


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.colors = list((
            {'id': 'red', 'color': (125, 15, 30)},
            {'id': 'orange', 'color': (180, 85, 40)},
            {'id': 'yellow', 'color': (210, 180, 30)},
            {'id': 'green', 'color': (30, 115, 80)},
            {'id': 'blue', 'color': (10, 65, 110)},
            {'id': 'violet', 'color': (45, 60, 100)})
        )

        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DetectFrame = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[919.62841,0,650.49677],[0,918.114,379.40409],[0,0,1]])
        self.extrinsic_matrix = np.eye(4)
        self.projection_matrix = np.array([[919.62841,0,650.49677,0],[0,918.114,379.40409,0],[0,0,1,0]])
        self.extrinsic_manual = np.array([[1,0,0,22],[0,-0.981,-0.191,226],[0,0.191,-0.981,980],[0,0,0,1]])
        self.distortion_matrix = np.array( [0.1645,-0.5088,-0.002298,0.0001067,0.4466])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        # self.grid_x_points = np.arange(-50,100,50)
        # self.grid_y_points = np.arange(-25,170,50)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points)).T.reshape(-1,2)
        self.tag_detections = np.array([])
        self.tag_uv_cam=np.float32([[-250, -25], [250, -25], [250, 275],[-250,275]])
        self.tag_uvd_cam=np.array([[-250, -25,0], [250, -25,0], [250, 275,0],[-250,275,0]])
        self.tag_d_cam=np.array([[0],[0],[0],[0]])
        self.tag_locations = np.array([[-250, -25,0], [250, -25,0], [250, 275,0],[-250,275,0]])
        self.tag_perspective_loc = np.float32([[380,550],[880,550],[880,250],[380,250]])
        self.perspectiveMat=np.eye(3)
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = (self.DepthFrameRaw-650) >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(cv2.warpPerspective(self.VideoFrame, self.perspectiveMat, (self.VideoFrame.shape[1], self.VideoFrame.shape[0])), (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(cv2.warpPerspective(self.DepthFrameRGB, self.perspectiveMat, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0])), self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None
        
    def convertQtDetectFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.DetectFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file

        """
        # yamlPath="/home/student_pm/armlabPro/calibration.yaml"
        cv_file=open(file)
        y=yaml.load(cv_file)
        # print(y)
        # print(y['camera_matrix']['data'])
        
        self.intrinsic_matrix=np.reshape(y['camera_matrix']['data'],(3,3))
        self.distortion_matrix=np.array(y['distortion_coefficients']['data'])
        self.projection_matrix=np.reshape(y['projection_matrix']['data'],(3,4))
        # self.cameraCalibrated=True
        pass

    def retrieve_area_color(self, data, cx, cy, labels):
        # mask = np.zeros(data.shape[:2], dtype="uint8")
        # cv2.drawContours(mask, [contour], -1, 255, -1)
        # print(data.shape)
        mean = cv2.mean(data[cy-5:cy+5,cx-5:cx+5])
        # print(mean)
        min_dist = (np.inf, None)
        for label in labels:
            d = np.linalg.norm(label["color"] - np.array(mean[0:3]))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        return min_dist[1] 


    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def max_Filter(self,input,size):
        output=np.zeros_like(input)
        sub_img=np.zeros((size,size))
        newInput = np.zeros((input.shape[0]+size-1,input.shape[1]+size-1))
        g=int((size-1)/2)
        newInput[g:input.shape[0]+g,g:input.shape[1]+g]=input
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                sub_img=newInput[i:i+size,j:j+size]
                sub_img=np.reshape(sub_img,-1)
                output[i][j]=np.max(sub_img)
        return output
    def detectBlocksInDepthImage(self):
        if np.sum(self.perspectiveMat)==3:
            depthImg = self.DepthFrameHSV[:,:,0].copy()
            detectRes = self.VideoFrame.copy()
        else:
            depthImg = cv2.warpPerspective(self.DepthFrameHSV[:,:,0], self.perspectiveMat, (self.DepthFrameHSV[:,:,0].shape[1], self.DepthFrameHSV[:,:,0].shape[0]))
            detectRes = cv2.warpPerspective(self.VideoFrame, self.perspectiveMat, (self.VideoFrame.shape[1], self.VideoFrame.shape[0]))
        mask = np.zeros_like(depthImg)
        cv2.rectangle(mask, (120,45),(1150,700), 255, cv2.FILLED)
        cv2.rectangle(mask, (550,450),(710,700), 0, cv2.FILLED)
        mask2 = np.zeros_like(depthImg)
        cv2.rectangle(mask2, (170,70),(1100,650), 255, cv2.FILLED)
        cv2.rectangle(mask2, (540,390),(720,700), 0, cv2.FILLED)
        # print(depthImg.shape,mask.shape)
        depthPro = cv2.bitwise_and(depthImg,mask)
        depthPro = cv2.medianBlur(depthPro,7)
        depthPro = cv2.Canny(depthPro,15,30)
        depthPro = cv2.bitwise_and(depthPro,mask2)
        # depthPro = self.max_Filter(depthPro,7)
        contours, _ = cv2.findContours(depthPro, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(detectRes, contours, -1, (0,255,255), thickness=1)
        # color=[]
        # theta=[]
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00']==0 or contourArea(contour)<=20:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            color = self.retrieve_area_color(detectRes, cx, cy, self.colors)
            theta = cv2.minAreaRect(contour)[2]
            cv2.putText(detectRes, color, (cx-30, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2)
        self.DetectFrame = detectRes

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_frame = self.VideoFrame.copy()
        if self.cameraCalibrated:
            EXT=self.extrinsic_matrix
        else:
            EXT=self.extrinsic_manual

        board_points_3D = np.column_stack((self.grid_points, np.zeros(self.grid_points.shape[0])))
        board_points_homogenous = np.column_stack((board_points_3D, np.ones(self.grid_points.shape[0])))

        camera_points = np.matmul(EXT, np.transpose(board_points_homogenous))
        pixel_locations = np.transpose(np.matmul(self.projection_matrix,camera_points))
     
        # for i in range(len(pixel_locations)):
        #     # pix_real = np.matmul(np.linalg.inv(self.perspectiveMat),np.array([pixel_locations[i][0],pixel_locations[i][1],0]).T)
        #     cv2.circle(modified_frame, (int(pixel_locations[i][0]/camera_points[2][i]+12), int(pixel_locations[i][1]/camera_points[2][i]-5)), 5,(0, 0, 255), 1)
        #     # cv2.circle(new_img, (int(pix_real[0]/camera_points[2][i]), int(pix_real[1]/camera_points[2][i])), 5,(0, 0, 255), 1)

        if np.sum(self.perspectiveMat)==3:
            new_img = modified_frame
        else:
            new_img = cv2.warpPerspective(modified_frame, self.perspectiveMat, (modified_frame.shape[1], modified_frame.shape[0]))
            # new_img = cv2.warpAffine(modified_frame, self.perspectiveMat, (modified_frame.shape[1], modified_frame.shape[0]))

        # if np.sum(self.perspectiveMat)==0:
        #     new_img = modified_frame
        # else:
        #     new_img = cv2.warpPerspective(modified_frame, self.perspectiveMat, (modified_frame.shape[1], modified_frame.shape[0]))
        # for i in range(len(pixel_locations)):
        #     pix_real = np.matmul(np.linalg.inv(self.perspectiveMat),np.array([pixel_locations[i][0],pixel_locations[i][1],0]).T)
        #     # cv2.circle(modified_frame, (int(pixel_locations[i][0]/camera_points[2][i]), int(pixel_locations[i][1]/camera_points[2][i])), 5,(0, 0, 255), 1)
        #     cv2.circle(new_img, (int(pix_real[0]/camera_points[2][i]), int(pix_real[1]/camera_points[2][i])), 5,(0, 0, 255), 1)

        
        self.GridFrame = new_img
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        # modified_image=np.ones(self.VideoFrame)
        modified_image = self.VideoFrame.copy()
        # Write your code here
        
        for item in msg.detections:
            idn=item.id
            pos=np.array([item.centre.x,item.centre.y])
            self.tag_uv_cam[idn-1]=[item.centre.x,item.centre.y]
            self.tag_uvd_cam[idn-1]=[item.centre.x,item.centre.y,self.DepthFrameRaw[int(item.centre.y),int(item.centre.x)]]
            self.tag_d_cam[idn-1]=np.array([self.DepthFrameRaw[int(item.centre.y),int(item.centre.x)]])
            corners=[[item.corners[0].x,item.corners[0].y],[item.corners[1].x,item.corners[1].y],[item.corners[2].x,item.corners[2].y],[item.corners[3].x,item.corners[3].y]]
            # print(pos,corners)
            cv2.circle(modified_image,(int(pos[0]),int(pos[1])),3,(0,255,0),-1)
            cv2.line(modified_image,(int(corners[0][0]),int(corners[0][1])),(int(corners[1][0]),int(corners[1][1])),(0,0,255),2)
            cv2.line(modified_image,(int(corners[1][0]),int(corners[1][1])),(int(corners[2][0]),int(corners[2][1])),(0,0,255),2)
            cv2.line(modified_image,(int(corners[2][0]),int(corners[2][1])),(int(corners[3][0]),int(corners[3][1])),(0,0,255),2)
            cv2.line(modified_image,(int(corners[3][0]),int(corners[3][1])),(int(corners[0][0]),int(corners[0][1])),(0,0,255),2)
            cv2.putText(modified_image,'ID='+str(idn),(int(pos[0])+20,int(pos[1])+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)


        if np.sum(self.perspectiveMat)==3:
            new_img = modified_image
        else:
            new_img = cv2.warpPerspective(modified_image, self.perspectiveMat, (modified_image.shape[1], modified_image.shape[0]))
            # new_img = cv2.warpAffine(modified_image, self.perspectiveMat, (modified_image.shape[1], modified_image.shape[0]))
        self.TagImageFrame = new_img

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        # self.camera.VideoFrame = cv2.warpPerspective(cv_image, self.camera.perspectiveMat, (cv_image.shape[1], cv_image.shape[0]))


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = cv2.warpPerspective(cv_depth, self.camera.perspectiveMat, (cv_depth.shape[1], cv_depth.shape[0]))
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Detect window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                grid_frame = self.camera.convertQtGridFrame()
                detect_frame = self.camera.convertQtDetectFrame()
                self.camera.projectGridInRGBImage()
                self.camera.detectBlocksInDepthImage()
                
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame, detect_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    # self.camera.VideoFrame = cv2.warpPerspective(cv_image, self.camera.perspectiveMat, (cv_image.shape[1], cv_image.shape[0]))
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Detect window",
                        cv2.cvtColor(self.camera.DetectFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()