# in Spyder: %matplotlib qt



import pyximport
pyximport.install()

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit


# Generate a list of n non-repeating random integers, arranged from smallest to largest

def get_random_integers(int n) -> list[int]:
    random_integers = random.sample(range(1, 100), n)
    random_integers.sort()
    return random_integers




# Calculate the strength of the repulsion, c1=1, c2=1

cdef double f(double r, double c1=1, double c2=1):
    # Calculate the force
    cdef double force
    force = c1 * math.exp(-r / c2)
    return force



# Calculate the total force of repulsion for each particle

def combined_force(p, int n) -> list[float]:
    cdef total_force = []
    cdef int i, j
    cdef double r, fn_sum, fn
    for i in range(n):
        fn_sum = 0
        for j in range(n):
            if j != i:
                r = p[j] - p[i]
                fn = -1 * f(abs(r)) * r / abs(r)
                fn_sum += fn
        total_force.append(fn_sum)
    return total_force



# Calculate the displacement between two particles

def displacement( list total_force, double eta=1, double delta_t=1) -> list[float]:
    cdef displacement
    displacement = [f / eta * delta_t for f in total_force]
    return displacement



# Update the position of particles

def update_position(list p, list delta_r, double min_x=0, double max_x=10) -> list[float]:

#    new_p = [p[i] + delta_r[i] for i in range(len(p))]

    cdef list new_p = []
    cdef int i
    cdef double new_xpos
    for i in range(len(p)):
        new_xpos = p[i] + delta_r[i]
        if new_xpos > max_x:
            new_xpos = max_x
        elif new_xpos < min_x:
            new_xpos = min_x
        new_p.append(new_xpos)
    return new_p



# Main function, Calculate the position of the particle after n time steps

def simulate(int n, int time_step, show_plot=False):

    # p = get_random_integers(n)
    cdef int x
    cdef list p = [x/2 for x in range(n)]
    

    print("P(0): ", p)
    cdef list total_force
    cdef list x_det
    cdef list xpos
    for i in range(time_step):
        total_force = combined_force(p, n)

        # print('total_force', total_force)

        x_det = displacement(total_force, delta_t=0.1)

        # print('x_det', x_det)

        p = update_position(p, x_det)

        xpos = p

        if show_plot:

            if i % 2 == 0:

                update_plot(xpos)

                print("in iteration", i)

        # time.sleep(0.0001)

        # print(f"{i=}")

        # use random numbers to imitate time integration

        # xpos += 0.05 * (np.random.rand(len(xpos)) - 0.5)
    
    print("P({}): ".format(time_step), p)



# visualisation of the particle positions

def update_plot(list xpos):

    """Expects array-like vectors xpos and ypos, and plots them as positions of

    particles 2 dimensions.

    """

    plt.clf()

    ypos = [0 for i in range(len(xpos))]

    plt.plot(xpos, ypos, "o")

    plt.xlim(left=-1, right=11)

    plt.grid()

    plt.draw()



    plt.pause(0.0001)





# Run the simulation for 5 time_step with Number of particles n = 10

#simulate(10, 500, show_plot=False)

#%prun simulate(10, 500, show_plot=False)
