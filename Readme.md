# Trials of TensorFlow from a Computationalist

afq, November 2017

## Intro
This is a small collection of my trials learning TensorFlow. There's a lot of different examples out there that seem overly complex and aren't really using the intended idioms. The documentation itself is no exception. (devs: Add a wiki talk page!) The notebooks and utilities are the best code I can write after sifting through everything I've dredged up on the internet, while constantly pestering my dear friend E. H. at Brain.

## Impetus

The example datafile 'fractureplane.db' included in this repository are the results of a finite element simulationof steady state Stokes flow coded up in [FEniCS](fenicsproject.org) to sample the flow of a fluid around proppants. The database was written with my library [SimDataDB](https://bitbucket.org/afqueiruga/simdatadb/) that some of the notebooks require to load. (It's just sqlite3 with one table with fields Dp, h, n, and v.)

Each sample is a different random distribution of particles with different densities and fracture heights. One of these samples looks like this:

![Flow in a fracture around proppants](images/flowfield.png)

With this super simple prototype, we're just looking for a function that looks like this:
$$\bar{v} = f(\Delta p,h,n)$$
The "real" research will make these all vector arguments and include more parameters, and tackle a wider range of problems.

My goals are:

1. Designing models that fit physical phenomena with emergent nonlinearities and phase boundaries.
2. Higher order methods for training (Newton's, Jacobian-free nonlinear CG, etc.)
3. Turn the trained model into something that can be used in a scientific code.
4. Check the in-production usage against the original training data and requesting new data points.

## Progress

1. [polynomials.ipynb](polynomials.ipynb) : Fitting the given data to a basis of polynomials
2. [model_loading.ipynb](model_loading.ipynb): Loading the saved model, freezing the variables, and rewriting a trimmed down graph.
3. [hessian_mnist.ipynb](hessian_mnist.ipynb): A side track investigating the hessian matrix of the softmax mnist model.

The file [afqstensorutils.py](afqstensorutils.py) contains all of the utility functions I'm writing.

The scripts [start_docker.sh](start_docker.sh) and [start_venv.sh](start_venv.sh) are how I load up the TensorFlow environment on my Mac (with docker) and my Linux box (with a virtualenv install), respectively. 
