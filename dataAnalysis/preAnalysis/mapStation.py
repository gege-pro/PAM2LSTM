# envs:keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import basemap

plt.figure(1)
map = basemap()
map.drawcoastlines()
plt.title(r'$World Map$', fontsize=24)
plt.show()