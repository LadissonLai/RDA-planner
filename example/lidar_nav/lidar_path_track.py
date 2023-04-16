from ir_sim.env import EnvBase
import sys
import numpy as np
from RDA_planner.mpc import MPC
from collections import namedtuple
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# environment
env = EnvBase('lidar_path_track.yaml', save_ani=False, display=True, full=False)
car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce')
obs = namedtuple('obstacle', 'center radius vertex cone_type velocity')

# saved ref path
npy_path = sys.path[0] + '/path_track_ref.npy'
ref_path_list = list(np.load(npy_path, allow_pickle=True))
env.draw_trajectory(ref_path_list, traj_type='-k') # plot path


def scan_box(state, scan_data):

    ranges = np.array(scan_data['ranges'])
    angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

    point_list = []
    obstacle_list = []

    for i in range(len(ranges)):
        scan_range = ranges[i]
        angle = angles[i]

        if scan_range < ( scan_data['range_max'] - 0.01):
            point = np.array([ [scan_range * np.cos(angle)], [scan_range * np.sin(angle)]  ])
            point_list.append(point)

    if len(point_list) < 4:
        return obstacle_list

    else:
        point_array = np.hstack(point_list).T

        dbscan = DBSCAN(eps=1.5, min_samples=4)
        labels = dbscan.fit_predict(point_array)

    
        rect = cv2.minAreaRect(point_array.astype(np.float32))
        box = cv2.boxPoints(rect)

        vertices = box.T

        trans = state[0:2]
        rot = state[2, 0]
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        global_vertices = trans + R @ vertices

        obstacle_list.append(obs(None, None, global_vertices, 'Rpositive', 0))

        return obstacle_list


        # obs_list : obstacle: (center, radius, vertex, cone_type, velocity)

def main():
    
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.shape[2], [10, 1], [10, 0.5])
    
    obstacle_template_list = [{'edge_num': 3, 'obstacle_num': 0, 'cone_type': 'norm2'}, {'edge_num': 4, 'obstacle_num': 1, 'cone_type': 'Rpositive'}] # define the number of obstacles in advance

    mpc_opt = MPC(car_tuple, ref_path_list, receding=10, sample_time=env.step_time, process_num=4, iter_num=5, obstacle_template_list=obstacle_template_list, obstacle_order=True, min_sd=0.5)
    
    for i in range(500):   
        
        obs_list_ref = env.get_obstacle_list()
        scan_data = env.get_lidar_scan()
        # obs_list : obstacle: (center, radius, vertex, cone_type, velocity)
        obs_list = scan_box(env.robot.state, scan_data)

        if len(obs_list) > 0: temp = plt.plot(obs_list[0].vertex[0,:], obs_list[0].vertex[1,:], 'b-')
        
        opt_vel, info = mpc_opt.control(env.robot.state, 4, obs_list)
        env.draw_trajectory(info['opt_state_list'], 'r', refresh=True)

        env.step(opt_vel, stop=False)
        env.render(show_traj=True, show_trail=True)

        if len(obs_list) > 0: temp.pop(0).remove()

        if env.done():
            env.render_once(show_traj=True, show_trail=True)
            break

        if info['arrive']:
            print('arrive at the goal')
            break

    env.end(ani_name='path_track', show_traj=True, show_trail=True, ending_time=10, ani_kwargs={'subrectangles':True})
    
if __name__ == '__main__':
    main()