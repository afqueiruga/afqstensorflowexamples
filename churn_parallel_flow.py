from fenics import *
from mshr import *
from afqsfenicsutil import write_vtk_f
from simdatadb import *

import multiprocessing as multi
import itertools

krylov_method = "gmres"

def make_a_mesh(h, L, n):
    box = Box(Point(-L/2.0, -L/2.0, -h/2.0), Point( L/2.0, L/2.0, h/2.0, ))

    buf = 0.01
    def place_a_proppant(R, h,L):
        z = np.random.uniform(-h/2.0+R+buf, h/2.0-R-buf)
        x = np.random.uniform(-L/2.0+R+buf,L/2.0-R-buf)
        y = np.random.uniform(-L/2.0+R+buf,L/2.0-R-buf)
        return Sphere(Point(x,y,z), R)
    N = int(n * h*L*L)
    R = 0.5
    print N
    proppants = [ place_a_proppant(R, h,L ) for i in range(N) ]
    domain = box
    for p in proppants:
        domain -= p
    mesh = generate_mesh(domain,16)

    # Now mark the boundary
    boundfunc = FacetFunction("size_t",mesh)
    boundfunc.set_all(0)
    everywhere = CompiledSubDomain("on_boundary",eps=1.0e-12)
    everywhere.mark(boundfunc,7)

    top = CompiledSubDomain(" x[2]>h/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    top.mark(boundfunc,1)
    bot = CompiledSubDomain("-x[2]>h/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    bot.mark(boundfunc,2)
    inlet = CompiledSubDomain("-x[0]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    inlet.mark(boundfunc,3)
    outlet = CompiledSubDomain("x[0]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    outlet.mark(boundfunc,4)
    sidewallA = CompiledSubDomain("x[1]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    sidewallA.mark(boundfunc,5)
    sidewallB = CompiledSubDomain("-x[1]>L/2.0-eps && on_boundary",h=h,L=L,eps=1.0e-12)
    sidewallB.mark(boundfunc,6)

    
    return mesh, boundfunc, (top,bot,inlet,outlet,sidewallA,sidewallB)

ocnt = 0
sdb = SimDataDB("./fractureplane.db")
sdb.Add_Table("flowrun",
              (("Dp","FLOAT"), ("h","FLOAT"), ("L","FLOAT"), ("n",'FLOAT') ),
              ( ("v","FLOAT"),) )
@sdb.Decorate("flowrun", memoize=False)
def solve_a_setup(Dp,h, L, n):
    global ocnt
    # Generate mesh
    mesh,bf,bcdoms = make_a_mesh(h,L,n)
    # File('mesh.pvd')<<mesh
    # Build function space
    V2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    S1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    S0 = FiniteElement("DG", mesh.ufl_cell(), 0)
    TH = V2 * S1
    W = FunctionSpace(mesh, TH)

    # Set up the boundary conditions
    noslip = Constant((0.0, 0.0, 0.0))
    bcs = [
        DirichletBC(W.sub(0),noslip,bf,7),
        DirichletBC(W.sub(0),noslip,bf,1),
        DirichletBC(W.sub(0),noslip,bf,2),
        DirichletBC(W.sub(1),0,bf,4),
        DirichletBC(W.sub(0).sub(1),0.0,bf,5),
        DirichletBC(W.sub(0).sub(1),0.0,bf,6)
    ]

    # Define variational problem
    (v,   p) = TrialFunctions(W)
    (tv, tp) = TestFunctions(W)
    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=bf)
    f = Constant((0.0, 0.0, 0.0))
    a = inner(grad(tv), grad(v))*dx + div(tv)*p*dx + tp*div(v)*dx
    L = inner(tv, f)*dx \
      - inner(tv, Constant(Dp)*n)*ds(3)
    b = inner(grad(tv), grad(v))*dx + tp*p*dx

    # Solve it
    U = Function(W)
    A, bb = assemble_system(a, L, bcs)
    P, btmp = assemble_system(b, L, bcs)
    solver = KrylovSolver(krylov_method, "amg")
    solver.set_operators(A, P)
    solver.solve(U.vector(),bb)
    #solve(a==L,U,bcs)
    v, p = U.split()
    
    # Post processing
    p_v = [ inner(v,n)*ds(i) for i in (3,4,1,2,5,6) ]
    print "Check: \int v.n = ", sum([ assemble(f) for f in p_v ])
    
    # Save solution in VTK format
    # u, p = U.split(deepcopy=True)
    # write_vtk_f("./viz_{0}.vtk".format(ocnt),mesh,
    #             {"p":p,"v":u})
    # ocnt += 1
    return tuple([ assemble(f) for f in (p_v[0],) ])

import numpy as np
hs = np.linspace(3.1,14.9,4)
Dps = [ -1.0 ]
Ls = [ 5.0 ] #np.linspace(5,15,5)
ns = np.linspace(0.01,0.2,14)#[ 0.0, 0.05,0.1 ]
def f(t):
    print t
    for i in xrange(6):
        solve_a_setup(*t)
    return 0
poo = multi.Pool(processes=2)
poo.map(f, itertools.product( Dps,hs,Ls,ns) )
