import random

import math

import matplotlib.pyplot as plt

import numpy as np

import time


# main function
def simulate(n, time_step, show_plot=False):

    # random particles
    np.random.seed(0)
    
    p = np.random.rand(n, 2)
    
# Reducing 2D to 1D    p = set_second_column_to_zero(p)

    print("P(0):\n", p)

    # time step loop
    for i in range(time_step):
        
        print("time step", i)
        
        # calculate total force for each particle
        total_force = combined_force(p, n)
        
        print("total_force \n",total_force)
        
        # calculate displacement for each particle
        x_det = displacement(total_force, delta_t=0.001)
        
        #print("x_det \n",x_det)

        # update the particles
        p = update_position(p, x_det)
        
        pos = p
        
        #print("pos \n",pos)
        
        # plot particles
        if show_plot:

            if i % 2 == 0:

                update_plot(pos,colors)

                print("in iteration", i)
            
    # plot finally result
    print("P({}): ".format(time_step), p)



# Reducing 2 dimensions to 1 dimension for model checking
def set_second_column_to_zero(matrix):

    matrix[:, 1] = 0
    
    return matrix


# Calculate the strength of the repulsion
def f(r, c1=1, c2=1):

    # Calculate the force
    
    abs_r = np.abs(r)
    
    #print("abs_r",abs_r)

    mag = c1 * 2.7 ** (-abs_r / c2)

    #print("mag",mag)

    return mag


# Calculate the total force for each particle
def combined_force(p, n):
    
    total_force = np.zeros_like(p)
    
    for i in range(n):
        
        fn_sum = np.zeros(2)
        
        for j in range(n):
            
            if j != i:
                
                r = p[j] - p[i]
                
                #print("r",r)
                #print("sign",np.sign(r))
                
                fn =  -1 * f(r) * np.sign(r) 

                fn_sum += fn 
                
            total_force[i] = fn_sum
            
    return total_force



# Calculate the displacement between two particles
def displacement(total_force, eta=1, delta_t=1):

    displacement = total_force / eta * delta_t

    return displacement



# Update the position of particles
def update_position(p, delta_r, min_x=0, max_x=1):
    
    new_pos = p + delta_r
    
    x_out_of_bounds = np.logical_or(new_pos[:,0] > max_x, new_pos[:,0] < min_x)
    
    y_out_of_bounds = np.logical_or(new_pos[:,1] > max_x, new_pos[:,1] < min_x)
    
    new_pos[x_out_of_bounds, 0] = np.clip(new_pos[x_out_of_bounds, 0], min_x, max_x)
    
    new_pos[y_out_of_bounds, 1] = np.clip(new_pos[y_out_of_bounds, 1], min_x, max_x)
    
    return new_pos


# Plot
def update_plot(pos,color):

    plt.clf()

    xpos = pos[:, 0]
    
    ypos = pos[:, 1]

    for i in range(len(colors)):
        plt.plot(xpos[i], ypos[i], "o", color=colors[i])

    plt.xlim(left=-0.1, right=1.1)
    
    plt.ylim(bottom=-0.1, top=1.1)

    plt.grid()

    plt.draw()

    plt.pause(0.0001)



# Example usage:
# colors = [random.choice(['r', 'g', 'b', 'y', 'm']) for _ in range(n)]
colors = ['red', 'green', 'blue', 'orange']  
    
simulate(n=4, time_step=1000, show_plot=True)
