"""
Kolmogorov flow test

Usage:
    kturb.py [--mesh=<mesh>] <config_file>

Options:
    --mesh=<mesh>              processor mesh (you're in charge of making this consistent with nproc) [default: None]

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

args = docopt(__doc__)
mesh = args['--mesh']
if mesh == 'None':
    mesh = None
else:
    mesh = [int(i) for i in mesh.split(',')]

logger = logging.getLogger(__name__)

runconfig = ConfigParser()
config_file = pathlib.Path(sys.argv[-1])
runconfig.read(str(config_file))
logger.info("Using config file {}".format(config_file))

# parameters
params = runconfig['params']
nL = params.getint('nL')
nx = params.getint('nx')
ny = params.getint('ny')
Re_sc = params.getfloat('Re_sc')

Re = np.sqrt(2) * Re_sc
Ω = nL*Re
logger.info("Re/Re_c = {:e}".format(Re_sc))
logger.info("Re = {:e}".format(Re))

# always on square domain
L = 2 * np.pi
x = de.Fourier('x',nx, interval=[0, L], dealias=3/2)
y = de.Fourier('y',ny, interval=[0, L], dealias=3/2)

domain = de.Domain([x,y], grid_dtype='float', mesh=mesh)

variables = ['u', 'v', 'p']

problem = de.IVP(domain, variables=variables)
problem.parameters['L'] = L
problem.parameters['Re'] = Re
problem.parameters['Ω'] = Ω
problem.parameters['nL'] = nL
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"

# Navier-Stokes
problem.add_equation("dt(u) - Lap(u)/(Ω) + dx(p) = - u*dx(u) - v*dy(u) + nL**2*cos(nL*y)/Ω")
problem.add_equation("dt(v) - Lap(v)/(Ω) + dy(p) = - u*dx(v) - v*dy(v)")

#incompressibility
problem.add_equation("dx(u) + dy(v) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

run_opts = runconfig['run']
dt = run_opts.getfloat('dt')

if run_opts.getfloat('stop_wall_time'):
    solver.stop_wall_time = run_opts.getfloat('stop_wall_time')
else:
    solver.stop_wall_time = np.inf

if run_opts.getint('stop_iteration'):
    solver.stop_iteration = run_opts.getint('stop_iteration')
else:
    solver.stop_iteration = np.inf

if run_opts.getfloat('stop_sim_time'):
    solver.stop_sim_time = run_opts.getfloat('stop_sim_time')
else:
    solver.stop_sim_time = np.inf


basedir = pathlib.Path('scratch')
outdir = "kturb_" + config_file.stem
data_dir = basedir/outdir
if domain.dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir()

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(data_dir/'checkpoints', wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(data_dir/'snapshots', sim_dt=5, max_writes=200)
snap.add_task("dx(v) - dy(u)", name='vorticity')
snap.add_task("u")
snap.add_task("v")
snap.add_task("p")
snap.add_task("u",name="uc", layout="c")
snap.add_task("v",name="vc", layout="c")
analysis_tasks.append(snap)

timeseries = solver.evaluator.add_file_handler(data_dir/'timeseries', iter=100)
timeseries.add_task("0.5*integ(u**2 + v**2)/L**2", name='Ekin')
timeseries.add_task("0.5*integ(u**2)/L**2", name='Ekin_x')
timeseries.add_task("0.5*integ(v**2)/L**2", name='Ekin_y')
analysis_tasks.append(timeseries)

# initial conditions
xx, yy = domain.grids(scales=domain.dealias)
phi = domain.new_field()
u = solver.state['u']
v = solver.state['v']

for f in [u, v, phi]:
    f.set_scales(domain.dealias, keep_data=False)


seed = None
shape = domain.local_grid_shape(scales=domain.dealias)
rand = np.random.RandomState(seed)

filter_frac = 0.1
ampl  = 1e-5

phi['g'] = ampl * rand.standard_normal(shape)
filter_field(phi,frac=filter_frac)

u['g'] = np.cos(nL*yy)/Ω + phi.differentiate('y')['g']
v['g'] = -phi.differentiate('x')['g']
# k1 = 10
# k2 = 13
# u['g'] = np.cos(yy) + 1e-3*(np.cos(k1*yy) + np.cos(k2*yy))
# v['g'] = 1e-3*(np.cos(k1*xx) + np.cos(k2*xx))

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("0.5*integ(u**2 + v**2)/L**2", name="ekin")

CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=0.3,
                     max_change=1.5, min_change=0.5)
CFL.add_velocities(('u', 'v'))
dt = CFL.compute_dt()

start  = time.time()
while solver.ok:
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}; Time = {:e}; dt = {:e}".format(solver.iteration, solver.sim_time, dt))
        logger.info("Total Ekin = {:10.7e}".format(flow.max("ekin")))
    solver.step(dt)
    dt = CFL.compute_dt()
stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
