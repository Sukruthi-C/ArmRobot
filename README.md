# Intelligent Robotic Manipulation: Vision-Guided Precision in Automated Systems
This project focuses on advancing the autonomy of robotic systems through the integration of enhanced computer vision and precise kinematic control. Leveraging the power of an RGB-D camera paired with the ROS 2 ecosystem, we developed a system capable of sophisticated block manipulation tasks. The project encompasses the calibration of vision sensors to achieve high-fidelity depth perception, formulation of forward and inverse kinematic models for accurate trajectory planning, and the implementation of autonomous task execution algorithms. The robotic arm successfully demonstrated stacking, alignment, and color-based segregation with a high degree of precision. Challenges such as optimizing vision-based algorithms and hardware-software synergy were addressed, enhancing the robot's operational effectiveness. Future enhancements aim to refine path planning algorithms and expand the kinematic model's capabilities, ultimately pushing the boundaries of what autonomous robotic systems can achieve in dynamic environments.


![Alt text](/media/image_blocks.png)


**Table of content**
- [Code structure](#code-structure)
- [How to start](#how-to-start)

## Code structure

### Relavent 
You do need to modify **some** of these files.
- [install_scripts](install_scripts)
    - [install_scripts/config](install_scripts/config)
        - `rs_l515_launch.py` - to launch the camera
        - `tags_Standard41h12.yaml` - to define the april tags you used on the board
    - `install_Dependencies.sh` - to install ROS2/All the ROS wrappers/Dependencies
    - `install_Interbotix.sh` - to install arm related stuff
    - `install_LaunchFiles.sh` - to move the files under `/config` to where it should to be 
- [launch](launch) - to store the launch files, details in [here](code/launch/README.md)
- [src](src) - where you actually write code
    - `camera.py` - Implements the Camera class for the RealSense camera. 
        - Functions to capture and convert frames
        - Functions to load camera calibration data
        - Functions to find and perform 2D transforms
        - Functions to perform world-to-camera and camera-to-world transforms
        - Functions to detect blocks in the depth and RGB frames
    - `control_station.py`
         - This is the main program. It sets up the threads and callback functions. Takes flags for whether to use the product of exponentials (PoX) or Denabit-Hartenberg (DH) table for forward kinematics and an argument for the DH table or PoX configuration. You will upgrade some functions and also implement others according to the comments given in the code.
    - `kinematics.py` - Implements functions for forward and inverse kinematics
    - `rxarm.py` - Implements the RXArm class
        - Feedback from joints
        - Functions to command the joints
        - Functions to get feedback from joints
        - Functions to do FK and IK
        - A run function to update the dynamixiel servos
        - A function to read the RX200 arm config file
    - `state_machine.py` - Implements the StateMachine class
        - The state machine is the heart of the controller
- [config](config)
    - `rx200_dh.csv` - Contains the DH table for the RX200 arm
        - You will need to fill this in
    - `rx200_pox.csv` - Containes the S list and M matrix for the RX200 arm.
        - You will need to fill this in


### Irrelavent
Not need to touch these files.
- [media](media) - where we store media that used for README instructions
- [src/resource](code/src/resource) - where we store the additional files used in the project

## How to start?
1. Go to [/install_scripts](install_scripts) and following the `README.md` instructions
2. Go to [/launch](launch) to start the ROS2 nodes with the `.sh` files following the `README.md` instructions
