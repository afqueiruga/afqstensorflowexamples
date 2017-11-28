# Trials of TensorFlow from a Computationalist
## Intro
This is a small collection of my trials learning TensorFlow. There's a lot of different examples out there that seem overly complex and aren't really using the intended idioms. The documentation itself is no exception. The notebooks and utilities are the best code I can write after sifting through everything I've dredged up on the internet, while constantly pestering my dear friend E. H. at Brain.

## Impetus


The example datafile 'fractureplane.db' included in this repository are the results of a finite element simulation to sample the flow of a fluid around proppants. Each sample is a different random distribution of particles with different densiteis and fracture heights.

My goals are:

1. Designing models that fit physical phenomena with emergent nonlinearities and phase boundaries.
2. Higher order methods for training.
3. Turn the trained model into something that can be used in a scientific code. 
4. Check your in-production usage against the original training data and requesting new data points.

The file afqstensorutils.py contains all of the utility functions I'm writing.
