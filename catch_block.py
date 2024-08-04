import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

n_trials = 999
incline = math.pi/6
del_t = 0.02
t_stop = 5
mass_train = 100
acc_grav = 9.81
width = 120
height = width*math.tan(incline)

kp = 300
ki = 10
kd = 300

time_arr = np.arange(0 , t_stop + del_t , del_t)
time_len = len(time_arr)

err = np.zeros((n_trials , time_len))
err_dot = np.zeros((n_trials , time_len))
err_int = np.zeros((n_trials , time_len))

initial_train_pos = width / math.cos(incline)
train_pos = np.zeros((n_trials , time_len))
train_pos_x = np.zeros((n_trials , time_len))
train_pos_y = np.zeros((n_trials , time_len))

cube_pos_x = np.zeros((n_trials , time_len))
cube_pos_y = np.zeros((n_trials , time_len))

initial_train_vel = 0
train_vel = np.zeros((n_trials , time_len))
train_vel_x = np.zeros((n_trials , time_len))
train_vel_y = np.zeros((n_trials , time_len))

force_applied = np.zeros((n_trials , time_len))
force_net = np.zeros((n_trials , time_len))
force_gravity = mass_train * acc_grav
force_gravity_tan = force_gravity * math.sin(incline)
force_gravity_rad = force_gravity * math.cos(incline)


def generate_random_coordinate(incline):
    rand_x = random.uniform(0,width)
    rand_y = random.uniform(width * math.tan(incline) + 20 + 6.5 , width * math.tan(incline) + 40 + 6.5)
    return (rand_x , rand_y)


for i in range(n_trials):
    catch = 0
    (cube_pos_x[i][0] , cube_pos_y[i][0]) = generate_random_coordinate(incline)
    for j in range(time_len):
        if j == 0:
            train_pos[i][0] = initial_train_pos
            train_pos_x[i][0] = train_pos[i][0] * math.cos(incline)
            train_pos_y[i][0] = train_pos[i][0] * math.sin(incline) + 6.5
            
            train_vel[i][0] = initial_train_vel
            train_vel_x[i][0] = 0
            train_vel_y[i][0] = 0

            err[i][0] = cube_pos_x[i][0] - train_pos_x[i][0]
            err_dot[i][0] = 0
            err_int[i][0] = 0

            force_applied[i][0] = kp * err[i][0]
            force_net[i][0] = force_applied[i][0] - force_gravity_tan

        else:
            cube_pos_x[i][j] = cube_pos_x[i][j-1]
            cube_pos_y[i][j] = cube_pos_y[i][0] - acc_grav * (time_arr[j] ** 2) / 2
            err[i][j] = cube_pos_x[i][j] - train_pos_x[i][j-1]

            err_dot[i][j] = (err[i][j] - err[i][j-1])/del_t
            err_int[i][j] = err_int[i][j-1] + (err[i][j] + err[i][j-1])*del_t/2

            force_applied[i][j] = kp * err[i][j] + ki * err_int[i][j] + kd * err_dot[i][j]
            force_net[i][j] = force_applied[i][j] - force_gravity_tan

            train_vel[i][j] = train_vel[i][j-1] + (1/mass_train)*(force_net[i][j] + force_net[i][j-1])*del_t/2
            train_vel_x[i][j] = train_vel[i][j] * math.cos(incline)
            train_vel_y[i][j] = train_vel[i][j] * math.sin(incline)
            
            train_pos[i][j] = train_pos[i][j-1] + (train_vel[i][j] + train_vel[i][j-1])*del_t/2
            train_pos_x[i][j] = train_pos[i][j] * math.cos(incline)
            train_pos_y[i][j] = train_pos[i][j] * math.sin(incline) + 6.5
        if (train_pos_x[i][j] - 5 < cube_pos_x[i][j] + 3 and train_pos_x[i][j] + 5 > cube_pos_x[i][j] - 3) or catch == 1:
            if (train_pos_y[i][j] + 3 < cube_pos_y[i][j] - 2 and train_pos_y[i][j] + 8 > cube_pos_y[i][j] + 2) or catch == 1:                
                if catch == 0:
                    change = train_pos_x[i][j] - cube_pos_x[i][j]
                    catch = 1
                cube_pos_x[i][j] = train_pos_x[i][j] - change
                cube_pos_y[i][j] = train_pos_y[i][j] + 5
    initial_train_pos = train_pos[i][time_len - 1]
    initial_train_vel = train_vel[i][time_len - 1]



frame_amount = time_len * n_trials

def update_plot(num):

    platform.set_data([train_pos_x[int(num/time_len)][num-int(num/time_len)*time_len]-3.1,\
    train_pos_x[int(num/time_len)][num-int(num/time_len)*time_len]+3.1],\
    [train_pos_y[int(num/time_len)][num-int(num/time_len)*time_len],\
    train_pos_y[int(num/time_len)][num-int(num/time_len)*time_len]])

    cube.set_data([cube_pos_x[int(num/time_len)][num-int(num/time_len)*time_len]-1,\
    cube_pos_x[int(num/time_len)][num-int(num/time_len)*time_len]+1],\
    [cube_pos_y[int(num/time_len)][num-int(num/time_len)*time_len],\
    cube_pos_y[int(num/time_len)][num-int(num/time_len)*time_len]])

    return platform , cube , success , again


fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(4,3)


ax_main=fig.add_subplot(gs[0:3,0:2],facecolor=(0.9,0.9,0.9))
plt.xlim(0,width)
plt.ylim(0,width)
plt.xticks(np.arange(0,width+1,10))
plt.yticks(np.arange(0,width+1,10))
plt.grid(True)

rail = ax_main.plot([0,width],[5,height+5],'k',linewidth=6)
platform, = ax_main.plot([],[],'r',linewidth=18)
cube, = ax_main.plot([],[],'y',linewidth=14)

bbox_props_success = dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1.0)
success = ax_main.text(40,60,'',size='20',color='g',bbox=bbox_props_success)

bbox_props_again = dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)
again = ax_main.text(30,60,'',size='20',color='r',bbox=bbox_props_again)

pid_ani = animation.FuncAnimation(fig,update_plot,frames=frame_amount ,interval=20,repeat=False,blit=True)
plt.show()