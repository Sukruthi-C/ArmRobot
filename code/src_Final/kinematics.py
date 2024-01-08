"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.optimize import fsolve,minimize


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    for i in range(len(angle)):
        while angle[i] > np.pi:
            angle[i] -= 2 * np.pi
        while angle[i] <= -np.pi:
            angle[i] += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    # 
    A_final = np.eye(4)
    for i in range(5):
        # what is the use of link and its type?
        a, alpha, d, theta = dh_params[i]
        Ai = get_transform_from_dh(float(a), float(alpha), float(d), float(theta)+joint_angles[i] )
        A_final = A_final @ Ai # What should be the sequence of @
    return A_final



def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    rotz = np.array([[np.cos(theta), -np.sin(theta), 0, 0], 
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    transz = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]])
    tranx = np.array([[1, 0, 0, a], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    rotx = np.array([[1, 0, 0, 0], 
                     [0, np.cos(alpha), -np.sin(alpha), 0],
                     [0, np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 0, 1]])
    Ai = rotz @ transz @ tranx @ rotx

    return Ai


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    R = T[0:3,0:3]
    Trans = T[0:3,3]
    # print(R.shape)
    theta = np.arccos(R[2,2])
    psi = np.arccos(-R[2,0]/(np.sin(theta)))
    phi = np.arcsin(R[1,2]/np.sin(theta))
    return [theta,psi,phi]


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    return T[0:2,0:2]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def equ(variables,constants):
    t1, t2, t3 =variables
    a, b = constants
    f1 = t1+t2+t3-1.57
    f2 = 103.91+205.73*np.cos(t1+0.245)-200*np.sin(t1+t2)-174.15*np.sin(t1+t2+t3)-a
    f3 = 205.73*np.sin(t1+0.245)+200*np.cos(t1+t2)+174.15*np.cos(t1+t2+t3)-b
    return [f1,f2,f3]

def equ_vert(variables,constants):
    t1, t2, t3 =variables
    a, b = constants
    f1 = t1+t2+t3-1.05
    f2 = 103.91+205.73*np.cos(t1+0.245)-200*np.sin(t1+t2)-174.15*np.sin(t1+t2+t3)-a
    f3 = 205.73*np.sin(t1+0.245)+200*np.cos(t1+t2)+174.15*np.cos(t1+t2+t3)-b
    return [f1,f2,f3]

def equ_horizon(variables,constants):
    t1, t2, t3 =variables
    a, b = constants
    f1 = t1+t2+t3
    f2 = 103.91+205.73*np.cos(t1+0.245)-200*np.sin(t1+t2)-174.15*np.sin(t1+t2+t3)-a
    f3 = 205.73*np.sin(t1+0.245)+200*np.cos(t1+t2)+174.15*np.cos(t1+t2+t3)-b
    return [f1,f2,f3]

def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    joint_angles = np.zeros(5)
    # base link
    joint_angles[0] = -np.arctan2(pose[0],pose[1])
    # end_effector
    joint_angles[4] = joint_angles[0]

    a = pose[2]
    b = np.sqrt(pose[0]**2+pose[1]**2)
    joint_angles[1:4] = fsolve(equ,[0.2,0.2,1.2],args=([a,b]))
    return joint_angles


def IK_Vert(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    joint_angles = np.zeros(5)
    # base link
    joint_angles[0] = -np.arctan2(pose[0],pose[1])
    # end_effector
    joint_angles[4] = joint_angles[0]

    a = pose[2]
    b = np.sqrt(pose[0]**2+pose[1]**2)
    joint_angles[1:4] = fsolve(equ_vert,[0.2,0.2,1.2],args=([a,b]))
    return joint_angles

def IK_Horizon(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    joint_angles = np.zeros(5)
    # base link
    joint_angles[0] = -np.arctan2(pose[0],pose[1])
    # end_effector
    joint_angles[4] = 0

    a = pose[2]
    b = np.sqrt(pose[0]**2+pose[1]**2)
    joint_angles[1:4] = fsolve(equ_horizon,[0,0,0],args=([a,b]))
    return joint_angles