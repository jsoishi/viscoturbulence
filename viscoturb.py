"""
Viscoturbulence. 2D hydro + Oldroyd B

Usage:
    viscoturb.py [--mesh=<mesh>] <config_file>

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
Wi = params.getfloat('Wi')
eta = params.getfloat('eta')

if Re_sc:
    Re = np.selfqrt(2) * Re_sc * nL
else:
    Re = params.getfloat('Re') * nL
Wi = Wi/nL

logger.info("Re = {:e}".format(Re))
logger.info("Wi = {:e}".format(Wi))
logger.info("eta = {:e}".format( eta))

# always on square domain
#L = nL * 2 * np.pi
L = 2 * np.pi
x = de.Fourier('x',nx, interval=[0, L])#, dealias=3/2)
y = de.Fourier('y',ny, interval=[0, L])#, dealias=3/2)

domain = de.Domain([x,y], grid_dtype='float', mesh=mesh)

variables = ['u', 'v', 'p',  'lU11', 'U12', 'lU22']
#variables = ['u', 'v', 'p',  'lU11', 'U12', 'lU22', 'σ11', 'σ12', 'σ22']

problem = de.IVP(domain, variables=variables)
problem.parameters['L'] = L
problem.parameters['nL'] = nL
problem.parameters['eta'] = eta
problem.parameters['Re'] = Re
problem.parameters['Wi'] = Wi
problem.substitutions['U11'] = 'exp(lU11)'
problem.substitutions['U22'] = 'exp(lU22)'
problem.substitutions['sigma11'] = 'U11*U11'
problem.substitutions['sigma12'] = 'U11*U12'
problem.substitutions['sigma22'] = 'U12*U12 + U22*U22'
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['Div_sigma_x'] = "(dx(sigma11) + dy(sigma12))"
problem.substitutions['Div_sigma_y'] = "(dx(sigma12) + dy(sigma22))"

# Navier-Stokes
problem.add_equation("dt(u) - Lap(u)/(Re*(1+eta)) + dx(p) = 2*eta*Div_sigma_x/(Wi*Re*(1+eta)) - u*dx(u) - v*dy(u) + nL**2*cos(nL*y)/Re")
problem.add_equation("dt(v) - Lap(v)/(Re*(1+eta)) + dy(p) = 2*eta*Div_sigma_y/(Wi*Re*(1+eta)) - u*dx(v) - v*dy(v)")

#incompressibility
problem.add_equation("dx(u) + dy(v) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0)")

# conformation tensor evolution
# use Cholsky Decomposition
problem.add_equation("dt(lU11) - dx(u) = -u*dx(lU11) - v*dy(lU11) + U12*dy(u)/U11 - (1 - 1/U11**2)/Wi")
problem.add_equation("dt( U12)         = -u*dx( U12) - v*dy( U12) + U22**2/U11*dy(u) + U11*dx(v) + U12*dy(v) - U12*(1 + 1/U11**2)/Wi")
problem.add_equation("dt(lU22) - dy(v) = -u*dx(lU22) - v*dy(lU22) - U12*dy(u)/U11 + (U12**2/(U11**2 * U22**2) - 1 + 1/U22**2)/Wi")

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
outdir = "viscoturb_" + config_file.stem
data_dir = basedir/outdir
if domain.dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir()

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(data_dir/'checkpoints', wall_dt=3540, max_writes=50)
check.add_system(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(data_dir/'snapshots', sim_dt=2e-1, max_writes=200)
snap.add_task("dx(v) - dy(u)", name='vorticity')
snap.add_task("u")
snap.add_task("v")
snap.add_task("u",name="uc", layout="c")
snap.add_task("v",name="vc", layout="c")
snap.add_task("sigma11")
snap.add_task("sigma12")
snap.add_task("sigma22")
# snap.add_task("sigma11", name='sigma11_kspace', layout='c')
# snap.add_task("sigma12", name='sigma12_kspace', layout='c')
# snap.add_task("sigma22", name='sigma22_kspace', layout='c')
analysis_tasks.append(snap)

timeseries = solver.evaluator.add_file_handler(data_dir/'timeseries', iter=100)
timeseries.add_task("0.5*integ(u**2 + v**2)/L**2", name='Ekin')
timeseries.add_task("integ(sigma11 + sigma22)/L**2", name='Sigma')
analysis_tasks.append(timeseries)
# initial conditions
xx, yy = domain.grids(scales=domain.dealias)

phi = domain.new_field()

seed = None
shape = domain.local_grid_shape(scales=domain.dealias)
rand = np.random.RandomState(seed)

filter_frac = 0.1
ampl  = 1e-3

u = solver.state['u']
v = solver.state['v']

lU11 = solver.state['lU11']
U12 = solver.state['U12']
lU22 = solver.state['lU22']

for f in [u, v, phi, lU11, U12, lU22]:
    f.set_scales(domain.dealias, keep_data=False)

phi['g'] = ampl * rand.standard_normal(shape)
filter_field(phi,frac=filter_frac)

u['g'] = np.cos(nL*yy) + phi.differentiate('y')['g']
v['g'] = -phi.differentiate('x')['g']

Wi = nL*Wi
lU11['g'] = np.log(np.sqrt(1 + Wi**2/2 * np.sin(nL*yy)**2))
U12['g'] = -Wi/2 * np.sin(nL*yy)/np.exp(lU11['g'])
#lU22['g'] = np.log(np.sqrt(1 - (Wi/2 * np.sin(yy))**2/(1 + Wi**2/2 * np.sin(yy)**2)))
lU22['g'] = np.log(np.sqrt(1 - U12['g']**2))


flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("0.5*integ(u**2 + v**2)/L**2", name="ekin")
flow.add_property("integ(sigma11 + sigma22)/L**2", name="sigma")
flow.add_property("abs(integ(u,'x')/L)", name="Umax")

start  = time.time()
while solver.ok:
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}; Time = {:e}".format(solver.iteration, solver.sim_time))
        logger.info("Total Ekin = {:10.7e}".format(flow.max("ekin")))
        logger.info("Total Sigma = {:10.7e}".format(flow.max("sigma")))
        logger.info("|U|_max = {:10.7e}".format(flow.max("Umax")))
    solver.step(dt)
stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
