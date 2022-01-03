import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mypy.typeshed.stdlib.tkinter.constants import X, Y
from scipy.signal.windows import gaussian

fig = plt.figure()

ax = fig.add_subplot(122)
# Show matrix in two dimensions
ax.matshow(gaussian, cmap="jet")

ax = fig.add_subplot(122, projection="3d")
# Show three-dimensional surface
ax.plot_surface(X, Y, gaussian, cmap="jet")
plt.show()