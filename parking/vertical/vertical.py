import irsim
from gctl.curve_generator import curve_generator
import numpy as np
from collections import namedtuple
from RDA_planner.mpc import MPC

env = irsim.make(save_ani=False, display=True, full=False)

#########
# 1. Generate the reference path
#########
point1 = np.array([ [18.0], [8.75], [0]])
point2 = np.array([ [22.25], [7.0], [1.57] ])
point3 = np.array([ [22.25], [1.75], [1.57] ])
# print(env.robot.state[0:3])
point_list = [env.robot.state[0:3], point1, point2, point3]

cg = curve_generator()
ref_path_list = cg.generate_curve('reeds', point_list, 0.5, 5, include_gear=True) # 全局路径通过reeds shepp生成
env.draw_trajectory(ref_path_list, traj_type='-b')
    

#########
# 2. MPC control
#########
car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce dynamics')
robot_info = env.get_robot_info(0)
car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.wheelbase, [3, 1], [1, 0.5], 'acker')

mpc_opt = MPC(car_tuple, ref_path_list, sample_time=env.step_time, enable_reverse=True, max_edge_num=4, max_obs_num=7)

for i in range(500):   
    
    if np.linalg.norm(env.robot.state[0:2] - point2[0:2]) <= 1:
        # print('arrive point2')
        mpc_opt.update_parameter(max_sd=0.1, min_sd=0.1, slack_gain=1)

    obs_list = env.get_obstacle_list_obstructed()
    print(obs_list)
    opt_vel, info = mpc_opt.control(env.robot.state, 4, obs_list)

    env.draw_trajectory(info['opt_state_list'], 'r', refresh=True)

    env.step(opt_vel)
    env.render(0.01, show_traj=True, show_trail=True)
        
    if info['arrive']:
        print('arrive at the goal')
        break
    
    if env.done():
        print('done')
        break

env.end(ani_name='reverse_park', show_traj=True, show_trail=True, ending_time=10, ani_kwargs={'subrectangles':True})