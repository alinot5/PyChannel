# PyChannel
Python channelflow code with wall actuations for control in this paper: https://arxiv.org/abs/2301.12098

To run this code you need to first install Anaconda. Then download the environment.yml file and create an environment with it by running "conda env create -f environment.yml" in the directory where the environment.yml file is located. Alternatively, open the Anaconda navigator, click on the environments tab, click the import button, and then find the environment.yml file. This is a copy of the environment we ran to validate the code works. You may not need all of the packages in this environment. It is important to note that we tested loading this environment on a Mac and different version numbers may be required for other operating systems. If you are unable to install using the environment.yml file you will have to conda install the packages in that file manually.

Now that you have the conda environment setup, to run the code open Env.py in either the Slots or the Arbitrary directory. If you run this script it should run a short trajectory of the DNS. Refer to the code in the "if __name__ == '__main__':" portion to see how to load the environment and run it. The Slots directory actuations with two localized gaussian "slot jets". The Arbitrary directory allows the user to input any periodic boundary condtion. As an example we used a period 2 sin wave in x multiplied by a period 3 sin wave in z. The TestData directory contains the 25 initial conditions used for testing in the paper. If you do not modify the reset() function of the environment it will pull one of those 25 intial conditions.

The motivation for this project was to address 2 challengs: 1. There are no 3D pseudo-spectral channelflow codes available in python, which makes communication with control algorithms in python difficult (e.g. reinforcement learning) and 2. Most existing channelflow codes do not give the user full control over the wall actuations. Despite this code being written for control of 3D Couette flow with jets, we want to emphasize that this can be used as a DNS with any choice of wall-normal velocity at both walls. Furthermore, if there is interest, we can extend this repository to include plane Poiseuille flow and variable streamwise velocity and spanwise velocity boundary conditions.
