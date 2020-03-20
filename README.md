## Corona virus Slovakia

This repository contains Monte Carlo simulation tools for corona virus spread predictions

Setup:

`pip install -r requirements.txt`

Run:

`python main.py`

Computational resources used (simulation with a single city with 6 million inhabitants 
in 10 different populations):
- RAM: <500 MB
- CPU: 1.6 s per simulated day


### Populations:

Models revolve around populations which are represented by the PopulationBase class (`tools/population.py`), which:

- keeps track of health states of its members
- keeps track of when its members contracted the virus
- immunity of members towards the virus
- which members are still alive
- random selection of subsamples
- which members have recovered from the illness
- define interaction patterns for individuals (interactions withing households, 
  random interactions within a community, etc.)

See `examples/basic.py` for example usage

### Population centres:

Abstract geographically defined concentrations of human population (e.g. cities) and
simulate daily interactions among different populations. Can hold multiple populations with
different parameters and interaction patterns (e.g. children, working class, elderly people etc.).
A transmission matrix can be defined to simulate different transmission probabilities between different 
populations.
 
Defined in `tools/population_centre.py` 

The class stores results for each timestep.

See `examples/single_city.py` for example usage

### Virus:

Viruses are represented by subclasses of the Virus class, which define:

- mean period between infection and recovery
- standard deviation of the period between infection and recovery
- transmission probability of the given virus (R), which can depend on multiple factors
- mortality of the given virus

n.b. only a single virus can spread among populations since they do not keep track of the type of virus


### TODO:

##### General:

Need to tune the model parameters (R, random interaction distributions, transmission probabilities, ...). 
This can be done by tuning the parameters so that a medium sized single city model would match results of 
https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
and heuristically define social interaction patterns for typical Slovak populations.

##### Code development:
- add configuration files for tunable parameters
- add data readers for various sources of municipal data (town populations, migration matrixes, ...)
- prepare a model with all Slovak cities, towns and villages
- prepare plotting / result visualization for geographical simulations
- run on multiple CPU cores
