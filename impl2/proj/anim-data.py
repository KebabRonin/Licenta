import polars as pl, xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import psycopg2.pool as psycopg2_pool

file = r'ClimSim_low-res_grid-info.nc'
grid = xr.open_dataset(file,engine='netcdf4')

conns = psycopg2_pool.ThreadedConnectionPool(1, 3, dbname="Data", user="postgres", password="admin", host="localhost")
conn = conns.getconn()

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

from matplotlib.animation import FuncAnimation, PillowWriter
# Create the animation
fig, ax = plt.subplots()
m = Basemap(ax=ax, projection='robin',lon_0=165,resolution='c')
x,y = m(grid.lon,grid.lat)
c = 'ptend_q0002_25'

anim = FuncAnimation(fig, update, frames=1_000, blit=False)
plt.show()
# writergif = PillowWriter(fps=30)
# anim.save(f'{c}.gif', writer=writergif)
# try:
# 	plt.show()
# except:
# 	# To save the animation as a video file
# 	exit(0)
