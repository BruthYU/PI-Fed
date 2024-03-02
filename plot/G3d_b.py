from arrow_line import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

# 3D arrow
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)



# Create a function to compute the surface
def f(theta):
  x = theta[0]
  y = theta[1]
  # return  (0.8*y)**2 - x**2
  return x * np.exp(-x**2-y**2)

def gradient_f(theta):
    x = theta[0]
    y = theta[1]
    g_x = np.exp(-x**2-y**2) - 2*x**2*np.exp(-x**2-y**2)
    g_y = -2*x*y*np.exp(-x**2-y**2)
    return [g_x,g_y]

def line_fix_y(grad_x, x_0, z_0, x):
    return grad_x*(x-x_0) + z_0

def line_fix_x(grad_y, y_0, z_0, y):
    return grad_y*(y-y_0) + z_0



# Make a grid of points for plotting
x, y = np.mgrid[-2:2:41j, -2:2:41j]

# Create a figure with axes in 3d projection
fig1 = plt.figure(figsize = (4,4), dpi = 300)
ax1 = fig1.add_subplot(111, projection='3d')

plot_args = {'rstride': 2, 'cstride': 2,
             'linewidth': 0.01, 'antialiased': False,'cmap':"coolwarm",
             'vmin': -1, 'vmax': 1, 'edgecolors':'k'}

ax1.plot_surface(x, y, f([x,y]), **plot_args)
ax1.view_init(azim= 120, elev= 15)
ax1.set_xlabel('x')
ax1.set_ylabel('y')


# Compute Importance by Point (1.2, 0.4)
point = [0.1, 0.0]
z_0 = f(point)
grad = gradient_f(point)

z_along_x = line_fix_y(grad[0], point[0], z_0, point[0] - 0.5)
z_along_y = line_fix_x(grad[1], point[1], z_0, point[1] - 0.5)



# newax = fig1.add_axes(ax1.get_position(), projection='3d',
#                      xlim = ax1.get_xlim(),
#                      ylim = ax1.get_ylim(),
#                      zlim = ax1.get_zlim(),
#                      facecolor = 'none',)
# newax.view_init(azim= 75, elev= 15)

ax1.arrow3D(point[0], point[1] , z_0+0.01,
              -0.5, 0, z_along_x-z_0,
              mutation_scale=9,
              arrowstyle="simple",
              color = 'blue'
              )

ax1.arrow3D(point[0], point[1], z_0,
              0, -0.5, z_along_y - z_0,
              mutation_scale=9,
              arrowstyle="simple",
              color = 'red'

              )

#plot importance line


# newax.set_zorder(1)
# ax1.set_zorder(0)
# newax.set_axis_off()

plt.xticks(fontsize=8)

plt.savefig('a.pdf')
plt.show()

