from ir_sim.env import EnvBase
import sys
import numpy as np
from RDA_planner.mpc import MPC
from collections import namedtuple
import time

# environment
env = EnvBase('path_track.yaml', save_ani=False, display=True, full=False)
car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce')

# saved ref path
npy_path = sys.path[0] + '/path_track_ref.npy'
ref_path_list = list(np.load(npy_path, allow_pickle=True))
env.draw_trajectory(ref_path_list, traj_type='-k') # plot path

def main():
    
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.shape[2], [10, 1], [10, 0.5])
    
    obstacle_template_list = [{'edge_num': 3, 'obstacle_num': 4, 'cone_type': 'norm2'}, {'edge_num': 4, 'obstacle_num': 1, 'cone_type': 'Rpositive'}] # define the number of obstacles in advance

    mpc_opt = MPC(car_tuple, ref_path_list, receding=10, sample_time=env.step_time, process_num=4, iter_num=3, obstacle_template_list=obstacle_template_list, obstacle_order=True)
    
    for i in range(500):   
        
        obs_list = env.get_obstacle_list()
        opt_vel, info = mpc_opt.control(env.robot.state, 4, obs_list)
        env.draw_trajectory(info['opt_state_list'], 'r', refresh=True)

        env.step(opt_vel, stop=False)
        env.render(show_traj=True, show_trail=True)

        if env.done():
            env.render_once(show_traj=True, show_trail=True)
            break

        if info['arrive']:
            print('arrive at the goal')
            break

    env.end(ani_name='path_track', show_traj=True, show_trail=True, ending_time=10, ani_kwargs={'subrectangles':True})
    
if __name__ == '__main__':
    main()