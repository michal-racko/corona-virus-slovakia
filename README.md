## Corona virus Slovakia

This repository contains Monte Carlo simulation tools for corona virus spread predictions

Setup:

`pip install -r requirements.txt`

Run:

`python main.py`

#### Populations:

Models revolve around populations which are represented by the PopulationBase class (`tools/population`), which:

- keeps track of health states of its members
- keeps track of when its members contracted the virus
- immunity of members towards the virus
- which members are still alive
- random selection of subsamples
- which members have recovered from the illness

See `simulations/basic.py` for example usage

### Virus:

Viruses are represented by subclasses of the Virus class, which define:

- mean period between infection and recovery
- standard deviation of the period between infection and recovery
- transmission probability of the given virus (R), which can depend on multiple factors
- mortality of the given virus

n.b. only a single virus can spread among populations since they do not keep track of the type of virus
