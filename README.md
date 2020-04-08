## Corona virus Slovakia

This repository contains a Monte Carlo simulation for predictions of virus spread in the country

Setup:

`pip install -r requirements.txt`

Run simulation:

`python main.py -c <path-to-config>`

Plotting:

`python main_plotting.py -c <path-to-config>`

Body of the simulation is in `tools/simulation/population.py`

Simulation uses an OD matrix based on people movements which is considered sensitive
and cannot be disclosed to the public.
A mock OD matrix based purely on town distances, 
tuned to yield results as close as possible to the ones with the original OD matrix 
was generated.
To run the simulation with the mock data:

`python main.py -c config/mock_data.yml`