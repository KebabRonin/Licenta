import polars as pl, xarray as xr
from my_utils import *
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.stdout.reconfigure(encoding='utf-8')
offset = 0
file_size = 384 * 24

file = r'C:\Users\KebabWarrior\Desktop\Facultate\ClimSim\grid_info\ClimSim_low-res_grid-info.nc'
grid = xr.open_dataset(file,engine='netcdf4')

def mapplot(c, data):
	k = plt.figure()
	x = grid.lon
	y = grid.lat
	m = Basemap(projection='robin',lon_0=165,resolution='c')
	print(data.shape)
	m.drawcoastlines(linewidth=0.5)
	x,y = m(grid.lon,grid.lat)
	plot = plt.tricontourf(x,y,data[c].to_numpy().squeeze(),cmap='viridis',levels=14)
	m.colorbar(plot)
	plt.show()

import psycopg2.pool as psycopg2_pool
conns = psycopg2_pool.ThreadedConnectionPool(1, 10, dbname="Data", user="postgres", password="admin", host="localhost")
conn = conns.getconn()

# Define the initial plot
def init():
    pass

# Define the update function
def update(i):
	global c
	df = pl.read_database(f"select * from public.train where {(i)*384} <= sample_id_int and sample_id_int < {(i+1)*384}", connection=conn)
	ax.clear()
	# fig.clear()
	m.drawcoastlines(linewidth=0.5)
	plot = plt.tricontourf(x,y,df[c].to_numpy(),cmap='viridis',levels=14)
	# m.colorbar(plot, ax=ax)
	plt.title(c + ' t=' + str(i))
	# m.plot()
	return ax,

# Create the animation
fig = plt.figure()
ax = Axes3D(fig)
m = Basemap(ax=ax, projection='robin',lon_0=0)
x,y = m(grid.lon,grid.lat)
c = 'ptend_q0002_25'

anim = FuncAnimation(fig, update, frames=10, blit=False)

# writergif = PillowWriter(fps=30)
# anim.save(f'{c}.gif', writer=writergif)
try:
	plt.show()
except:
	# To save the animation as a video file
	exit(0)
