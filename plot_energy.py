import numpy as np
import sys
import pathlib
import h5py
import matplotlib.pyplot as plt

basedir = pathlib.Path(sys.argv[-1])
dfile = basedir / 'timeseries/timeseries_s1.h5'

data = h5py.File(str(dfile), "r")
sigma0 = 38.
KE0 = 0.25
t = data['scales/sim_time'][:]
KE = data['tasks/Ekin'][:,0,0] - KE0
#sigma = data['tasks/Σ'][:,0,0]
sigma = data['tasks/Sigma'][:,0,0] - sigma0
plt.subplot(121)
plt.semilogy(t, np.abs(KE))
plt.xlabel("time")
plt.ylabel("Kinetic energy")
plt.subplot(122)
# plt.plot(t, KE, 'x-')
# plt.xlim(0,500)
# plt.xlabel("time")
# plt.ylabel("Kinetic energy")

plt.semilogy(t, np.abs(sigma))
plt.xlabel("time")
plt.ylabel(r"$\Sigma$")

plt.savefig('energy.png')

