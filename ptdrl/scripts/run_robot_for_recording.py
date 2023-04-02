import numpy as np
import rospy
import task_env
import numpy as np
import random
from std_msgs.msg import Float32

# Random choose parameters and run simulations

pub_inflation = rospy.Publisher('/inflation', Float32, queue_size=10)

def rand_params():

    rand = np.random.rand(8)
    rand[0] = rand[0]*(1.59 - 0.1) + 0.1
    rand[1] = rand[1]*(1.5 - 0.5) + 0.5
    rand[2] = int(rand[2]*(20 - 5) + 5)
    rand[3] = int(rand[3]*(60 - 10) + 10)
    rand[4] = rand[4]*(0.6 - 0.01) + 0.01
    rand[5] = int(rand[5]*(32 - 16) + 16)
    rand[6] = int(rand[6]*(20 - 5) + 5)
    rand[7] = rand[7]*(1 - 0.1) + 0.1

    params = { 'max_vel_x' : rand[0], 'min_vel_x' : -rand[0], 'max_vel_trans': rand[0], 'max_vel_theta': rand[1], 'vx_samples': rand[2], 'vth_samples': rand[3], 'occdist_scale': rand[4],
        'path_distance_bias': rand[5], 'goal_distance_bias': rand[6], 'inflation_radius': rand[7]}
    
    return params

def rand_time():

    return random.randint(1,20)

def main():
    # Environment

    env = task_env.PtdrlTaskEnv()
    max_episodes = 500
    

    for i in range(max_episodes):
        print(f"Episode num: {i}")
        env.reset()
        done = False
        episode_steps = 0

        while not done:
            episode_steps += 1

            params = rand_params()
            #print(params)
            
            time = rand_time()
            #print(time)
            for i in range(time):
                inflation = Float32()
                inflation.data = params['inflation_radius']
                pub_inflation.publish(inflation)
                next_state, reward, done = env.step(params) # Step
                if done:
                    break

            episode_steps += 1

            # Failure if time is more than 400 timesteps
            if episode_steps >400:
                done = 1
        
        print(f"Timesteps: {episode_steps}")


if __name__ == '__main__':
    rospy.init_node('init_train')
    main()
