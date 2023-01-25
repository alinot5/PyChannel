# PyChannel
Python channelflow code with wall actuations for control

To run this code you need to first install Anaconda with Python 3.8. Then download the enironment.yml file and create an environment with it by running "conda env create -f environment.yml". Now that you have the environment setup, open Env.py in the Slots directory. If you run this script it should run a short trajectory of the DNS. Refer to the code in the "if __name__ == '__main__':" portion to see how to load the environment and run it. The TestData directory contains the 25 initial conditions used for testing in the paper. If you do not modify the reset() function of the environment it will pull one of those 25 intial conditions.
