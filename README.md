# PyChannel
Python channelflow code with wall actuations for control

To run this code you need to first install Anaconda. Then download the enironment.yml file and create an environment with it by running "conda env create -f environment.yml" in the directory where the environment.yml file is located. This is a copy of the environment we ran to validate the code works. You may not need all of the packages in this environment

Now that you have the conda environment setup, open Env.py in the Slots directory. If you run this script it should run a short trajectory of the DNS. Refer to the code in the "if __name__ == '__main__':" portion to see how to load the environment and run it. The TestData directory contains the 25 initial conditions used for testing in the paper. If you do not modify the reset() function of the environment it will pull one of those 25 intial conditions.

The motivation for this project was to address 2 challengs: 1. There are no 3D channelflow codes available in python, which makes communication with control algorithms in python difficult (e.g. reinforcement learning) and 2. Most existing channelflow codes do not give the user full conrol over the wall actuations. Despite this code being written for control of 3D Couette flow, we want to emphasize that this can be used as a DNS with any choice of wall-normal velocity at both walls. Furthermore, if there is interest, we can extend this repository to include plane Pouisellie flow and variable streamwise velocity and spanwise velocity boundary conditions.
