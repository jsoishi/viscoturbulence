"""
Viscoturbulence eigenvalue problem. 2D hydro + Oldroyd B

Stability of Kolmogorov flow

Usage:
    viscoturb_linear.py <config_file>

"""
import sys
import os
import time
import logging
import pathlib
import numpy as np

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools

from filter_field import filter_field

from configparser import ConfigParser
from docopt import docopt

logger = logging.getLogger(__name__)

runconfig = ConfigParser()
config_file = pathlib.Path(sys.argv[-1])
runconfig.read(str(config_file))
logger.info("Using config file {}".format(config_file))

# parameters
params = runconfig['params']
nL = params.getint('nL')
ny = params.getint('ny')
Wi = params.getfloat('Wi')
eta = params.getfloat('eta')
Re = params.getfloat('Re') * nL
kx = params.getfloat('kx')
Wi = Wi/nL

logger.info("Re = {:e}".format(Re))
logger.info("Wi = {:e}".format(Wi))
logger.info("eta = {:e}".format( eta))
logger.info("nL = {:}".format(nL))
logger.info("kx = {:}".format(kx))


# always on square domain
#L = nL * 2 * np.pi
L = 2 * np.pi
y = de.Chebyshev('y',ny, interval=[0, L])#, dealias=3/2)

domain = de.Domain([y], grid_dtype='float')

variables = ['u', 'v', 'p',  'σ11', 'σ12', 'σ22', 'u_y', 'v_y']

problem = de.EVP(domain, variables=variables, eigenvalue='γ')
problem.parameters['L'] = L
problem.parameters['nL'] = nL
problem.parameters['eta'] = eta
problem.parameters['Re'] = Re
problem.parameters['Wi'] = Wi
problem.parameters['kx'] = kx
problem.substitutions['u0'] = "cos(nL*y)"
problem.substitutions['u0_y'] = "-nL*sin(nL*y)"
problem.substitutions['σ11_0'] = "1 + Wi**2/2 * nL**2 * sin(nL*y)**2"
problem.substitutions['σ12_0'] = "-Wi/2 * nL * sin(nL*y)"
problem.substitutions['σ22_0'] = "1"
problem.substitutions['σ11_0_y'] = "Wi**2 * nL**3 * sin(nL*y)*cos(nL*y)"
problem.substitutions['σ12_0_y'] = "-Wi/2 * nL**2 * cos(nL*y)"
problem.substitutions['dx(A)'] = "1j*kx*A"
problem.substitutions['dt(A)'] = "γ*A"
problem.substitutions['Lap(A, A_y)'] = "dx(dx(A)) + dy(A_y)"
problem.substitutions['Div_σ_x'] = "(dx(σ11) + dy(σ12))"
problem.substitutions['Div_σ_y'] = "(dx(σ12) + dy(σ22))"


# Navier-Stokes
problem.add_equation("dt(u) - Lap(u, u_y)/(Re*(1+eta)) + dx(p) - 2*eta*Div_σ_x/(Wi*Re*(1+eta)) + u0*dx(u) + u0_y*v = 0")
problem.add_equation("dt(v) - Lap(v, v_y)/(Re*(1+eta)) + dy(p) - 2*eta*Div_σ_y/(Wi*Re*(1+eta)) + u0*dx(v) = 0")

#incompressibility
problem.add_equation("dx(u) + v_y = 0")

# conformation tensor evolution
problem.add_equation("dt(σ11) + u0*dx(σ11) + v*σ11_0_y - u0_y*σ11 - dx(u)*σ11_0 - dy(u)*σ12_0 - σ11_0*dx(u) - σ12_0*u_y + σ12*u0_y + 2*σ11/Wi = 0")
problem.add_equation("dt(σ12) + u0*dx(σ12) + v*σ12_0_y - u0_y*σ22 - dx(u)*σ12_0 - dy(u)*σ22_0 - σ11_0*dx(v) - σ12_0*dy(v) + 2*σ12/Wi = 0")
problem.add_equation("dt(σ22) + u0*dx(σ22) - dx(v)*σ12_0 - dy(v)*σ22_0 - σ12_0*dx(v) - σ22_0*dy(v) + 2*σ22/Wi = 0")

# First order
problem.add_equation("dy(u) - u_y = 0")
problem.add_equation("dy(v) - v_y = 0")

# Boundary Conditions
problem.add_bc("left(u) = right(u)")
problem.add_bc("left(v) = right(v)")
problem.add_bc("left(σ11) = right(σ11)")
problem.add_bc("left(σ12) = right(σ12)")
problem.add_bc("left(σ22) = right(σ22)")

# Build solver
solver = problem.build_solver()
logger.info('Solver built')

basedir = pathlib.Path('scratch')
outdir = "viscoturb_linear_" + config_file.stem
data_dir = basedir/outdir
if domain.dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
