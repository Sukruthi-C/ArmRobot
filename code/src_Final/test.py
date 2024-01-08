import cv2
import yaml
import numpy as np
import resource.config_parse
import sympy
from scipy.optimize import fsolve,minimize
import kinematics
# colors = list((
#             {'id': 'red', 'color': (125, 15, 30)},
#             {'id': 'orange', 'color': (180, 85, 40)},
#             {'id': 'yellow', 'color': (210, 180, 30)},
#             {'id': 'green', 'color': (30, 115, 80)},
#             {'id': 'blue', 'color': (10, 65, 110)},
#             {'id': 'violet', 'color': (45, 60, 100)})
#         )
# min_dist = (np.inf, None)
# for label in colors:
#     d = np.linalg.norm(label["color"])
#     if d < min_dist[0]:
#         min_dist = (d, label["id"])
# print(min_dist[1]=='violet')
file = "/home/student_pm/armlabPro/calibration.yaml"
cv_file=open(file)
y=yaml.load(cv_file)
# print(y)
# print(y['camera_matrix']['data'])

intrinsic_matrix=np.reshape(y['camera_matrix']['data'],(3,3))
distortion_matrix=np.array(y['distortion_coefficients']['data'])
projection_matrix=np.reshape(y['projection_matrix']['data'],(3,4))

print(intrinsic_matrix)
print(distortion_matrix)
print(projection_matrix)
print(np.rad2deg([ 0.31053707,  0.40179799, -1.32572065,  2.22132551,  0.31053707]))


# queue=list(({'color': 'red', 'num': 1},{'color': 'orange', 'num': 2},{'color': 'yellow', 'num': 3},{'color': 'green', 'num': 4},{'color': 'blue', 'num': 5},{'color': 'violet', 'num': 6}))
queue=['red','orange','yellow']
print(queue['orange'],queue['red'],queue['orange']<queue['red'])
# print(queue.index('red'))

