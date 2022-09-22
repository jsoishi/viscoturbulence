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
from dedalus.extras import flow_tools
import dedalus.public as d3

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
Re = params.getfloat('Re')
Wi = params.getfloat('Wi')
η = params.getfloat('eta')

logger.info("Re = {:e}".format(Re))
logger.info("Wi = {:e}".format(Wi))
logger.info("eta = {:e}".format(η))

# always on square domain
L = nL * 2 * np.pi
coords = d3.CartesianCoordinates('x', 'y')
dtype = np.float64
dealias = 3/2
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds = (0, L), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds = (0, L), dealias = dealias)
x = xbasis.local_grid(1)
y = ybasis.local_grid(1)

p = dist.Field(name='p', bases=(xbasis,ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
σ = dist.TensorField((coords, coords), name='σ', bases=(xbasis,ybasis))
tau_p = dist.Field(name='tau_p')

I = dist.TensorField((coords, coords), name='I')
I['g'][0][0] = 1
I['g'][1][1] = 1

cos_y = dist.VectorField(coords, bases=(xbasis, ybasis), name='cos_y')
cos_y['g'][0] = np.cos(y)

curl2D = lambda A: d3.skew(d3.grad(A))

problem = d3.IVP([p, u, σ, tau_p], namespace=locals())

# Navier-Stokes
# problem.add_equation("dt(u) - Lap(u)/(Re*(1+η)) + dx(p) = 2*η*Div_σ_x/(Wi*Re*(1+η)) - u*dx(u) - v*dy(u) + cos(y)/Re")
# problem.add_equation("dt(v) - Lap(v)/(Re*(1+η)) + dy(p) = 2*η*Div_σ_y/(Wi*Re*(1+η)) - u*dx(v) - v*dy(v)")

problem.add_equation("dt(u) - lap(u)/(Re*(1+η)) + grad(p) = 2*η*div(σ)/(Wi*Re*(1+η)) - dot(u, grad(u)) + cos_y/Re")
#incompressibility & gauge
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

# conformation tensor evolution
problem.add_equation("dt(σ) = - dot(u,grad(σ)) + dot(transpose(grad(u)), σ) + dot(σ, grad(u))- 2*(σ - I)/Wi")
# problem.add_equation("dt(lU11) - dx(u) = -u*dx(lU11) - v*dy(lU11) + U12*dy(u)/U11 - (1 - 1/U11**2)/Wi")
# problem.add_equation("dt( U12)         = -u*dx( U12) - v*dy( U12) + U22**2/U11*dy(u) + U11*dx(v) + U12*dy(v) - U12*(1 + 1/U11**2)/Wi")
# problem.add_equation("dt(lU22) - dy(v) = -u*dx(lU22) - v*dy(lU22) - U12*dy(u)/U11 + (U12**2/(U11**2 * U22**2) - 1 + 1/U22**2)/Wi")

# sigmas
# problem.add_equation("σ11 = U11*U11")
# problem.add_equation("σ12 = U11*U12")
# problem.add_equation("σ22 = U12*U12 + U22*U22")

# Build solver
solver = problem.build_solver(d3.RK222)
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
if dist.comm.rank == 0:
    if not data_dir.exists():
        data_dir.mkdir()

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler(data_dir/'checkpoints', wall_dt=3540, max_writes=50)
check.add_tasks(solver.state)
analysis_tasks.append(check)

snap = solver.evaluator.add_file_handler(data_dir/'snapshots', sim_dt=1e-1, max_writes=200)
snap.add_task(curl2D(u), name='vorticity')
snap.add_task(u)
snap.add_task(σ)

# snap.add_task("σ11", name='σ11_kspace', layout='c')
# snap.add_task("σ12", name='σ12_kspace', layout='c')
# snap.add_task("σ22", name='σ22_kspace', layout='c')
analysis_tasks.append(snap)

timeseries = solver.evaluator.add_file_handler(data_dir/'timeseries', iter=100)
timeseries.add_task(0.5*d3.integ(d3.dot(u,u))/L**2, name='Ekin')
timeseries.add_task(d3.integ(d3.trace(σ))/L**2, name='Σ')
analysis_tasks.append(timeseries)
# initial conditions
phi = dist.Field(name='phi', bases=(xbasis,ybasis))

ampl  = 1e-3
phi.fill_random('g', seed=42, distribution='normal', scale=ampl) # Random noise

u = curl2D(phi) + cos_y
σ['g'][0][0] = 1 + Wi**2/2 * np.sin(y)**2
σ['g'][0][1] = Wi/2 * np.sin(y)
σ['g'][1][0] = Wi/2 * np.sin(y)
σ['g'][1][1] = 1.
                    
# lU11['g'] = np.log(np.sqrt(1 + Wi**2/2 * np.sin(yy)**2))
# U12['g'] = -Wi/2 * np.sin(yy)/np.exp(lU11['g'])
# lU22['g'] = np.log(np.sqrt(1 - (Wi/2 * np.sin(yy))**2/(1 + Wi**2/2 * np.sin(yy)**2)))

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property(0.5*d3.integ(d3.dot(u,u))/L**2, name="ekin")
flow.add_property(d3.integ(d3.trace(σ))/L**2, name="sigma")

start  = time.time()
while solver.proceed:
    if (solver.iteration-1) % 10 == 0:
        logger.info("Step {:d}; Time = {:e}".format(solver.iteration, solver.sim_time))
        logger.info("Total Ekin = {:10.7e}".format(flow.max("ekin")))
        logger.info("Total Sigma = {:10.7e}".format(flow.max("sigma")))
    solver.step(dt)
stop = time.time()

logger.info("Total Run time: {:5.2f} sec".format(stop-start))
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
