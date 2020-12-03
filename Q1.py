import numpy as np

"Reservoir simulation exercise"

"""
Consider a 2D matrix that has the initial shape:
    
     x(m)   0  250  750  1250 1750 2000
     t(s) 
     
      0     170 200 200  200  200  200
    6.25    170  ()  ()   ()  ()   200
    
a = k/(rho x miu x c) = 5 m2/s
a_ = delta_t / delta_x_
b = (delta_t / delta_x) x (k/(tho x miu x c) = a_ x a
"""

#input Time and Distance

#LENGTH OF THE RESERVOIR
x = 2000 

#LENGTH OF ONE GRID
delta_x = 500

#TOTAL NUMBER OF GRIDS (WITHOUT BOUNDARY CONDITIONS) 
a = x/delta_x

#RESERVOIR CONSTANT
res_constant = 5

#STABILITY CRITERION
delta_t_max = 0.5 * (1/res_constant) * delta_x**2

#TIME STEP
delta_t = 0.25 * delta_t_max

#CONSTANT TERM IN EQUATION INCLUDING TIME AND DISTANCE
a_ = delta_t / delta_x**2

#CONSTANT TERM CONSISTING OF a_ AND RESERVOIR CONSTANT
b = a_ * res_constant

#INITIALIZE TIME 
time = [0]

#NUMBER OF ITERATION THROUGH TIME
iterations = 10
for i in range(iterations):
    time.append(time[i] + delta_t_max)

#EMPTY MATRIX INITIATION
iterations = int(len(time))      #number of iterations
cells = int(a + 2)          #number of cells a + 2 because we set the bundary conditions
P = np.zeros((iterations, cells))    #empty matrix of raws=time step and columns=cells

P[0,:] = 200            #set all the cells=200 on the first raw
P[:,int(a+1)] = 200     #set all the cells=200 on the far right column
P[:,0] = 170            #set all the cells=170 on the far left  column

#START BY INITIATING THE FIRST LOOP WHICH WILL LOOP THROUGH TOTAL NUMBER OF TIME STEPS - 1 
#BECAUSE THE FIRST RAW IS ALREADY DEFINED THROGUH INITIAL CONDITIONS

for i in range(iterations-1): 
    # NOW WE WANT TO DO FILL ALL THE CALCULATIONS FOR ONE TIME STEP AT ONCE
    for j in range(1):
        # WE CAN START WITH THE FIRST CELLS NEAR THE BOUNDARY CONDITIONS
        
        #LEFT CELL
        P[i+1,1] = P[i,1] + (4/3)*b*(P[i, 2] - 3*P[i,1] + 2*P[i, 0])
        
        #RIGHT CELL
        P[i+1, cells-2] = P[i,cells-2] + (4/3)*b*(P[i, cells-3] - 3*P[i,cells-2] + 2*P[i,cells-1])
        
        #FOR ALL THE CELLS IN BETWEEN
        #TECHNIQUE: WE WILL LOOP ON THE RAW, ONLY THOUGH THE MIDDLE CELLS, HENCE WE WILL EXCLUDE
        #4 CELLS, THE 2 BOUNDARY FIXED PRESSURES AND THE 2 CELLS NEAR BOUNDARY THAT WE CALCULATED
        #HENCE WE WILL NEED TO LOOP ONLY CELLS-4 TIMES
        for k in range(cells-4):
            P[i+1, k+2] = P[i,k+2] + b*( P[i, k+3] - 2*P[i, k+2] + P[i, k+1] )
            
            




"""---------IREGULAR SHAPE--------"""
    
import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D  # NOQA

spatial_axes = [1, 5, 1]
filled = np.ones(spatial_axes)

colors = np.empty(spatial_axes + [4], dtype=np.float32)
alpha = .5

colors[0] = 'blue'
colors[1] = 'blue'
colors[2] = [0, 0, 1, alpha]
colors[3] = [1, 1, 0, alpha]
colors[4] = [0, 1, 1, alpha]
colors[5] = [0, 1, 1, alpha]

# set all internal colors to black with alpha=1
"""colors[1:-1, 1:-1, 1:-1, 0:3] = 0
colors[1:-1, 1:-1, 1:-1, 3] = 1"""

fig = plt.figure()

ax = fig.add_subplot('111', projection='3d')

ax.voxels(filled, facecolors=colors, edgecolors='k')





"""---------IREGULAR SHAPE--------"""

import matplotlib.pyplot as plt

import numpy as np
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()



"""---------RAINBOW SHAPE--------"""

#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.size'] = 14

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# prepare some coordinates, and attach rgb values to each
r, g, b = np.indices((17, 17, 17)) / 16.0
rc = midpoints(r)
gc = midpoints(g)
bc = midpoints(b)


sphere = rc > -1

# combine the color components
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = rc
colors[..., 1] = gc
colors[..., 2] = bc

# and plot everything
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')
ax.voxels(r, g, b, sphere,
          facecolors=colors,
          edgecolors=np.clip(colors-1, 0, 0),  # black
          linewidth=0.5,
          )
ax.set(xlabel='R', ylabel='G', zlabel='B')
plt.savefig("3Dvoxel_rgb_cube_rgb000_axisequal.png", dpi=130,transparent = False)
plt.show()