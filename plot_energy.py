import sys
import pathlib
import h5py
import matplotlib.pyplot as plt

basedir = pathlib.Path(sys.argv[-1])
dfile = basedir / 'timeseries/timeseries.h5'

data = h5py.File(str(dfile), "r")

t = data['scales/sim_time'][:]
KE = data['tasks/Ekin'][:,0,0]
#sigma = data['tasks/Σ'][:,0,0]
#sigma = data['tasks/Sigma'][:,0,0]
plt.subplot(121)
plt.plot(t, KE)
plt.xlabel("time")
plt.ylabel("Kinetic energy")
plt.subplot(122)
plt.plot(t, KE, 'x-')
plt.xlim(0,500)
plt.xlabel("time")
plt.ylabel("Kinetic energy")

# plt.plot(t, sigma)
# plt.xlabel("time")
# plt.ylabel(r"$\Sigma$")

plt.savefig('energy.png')

