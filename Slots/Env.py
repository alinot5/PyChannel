#!/home/linot/anaconda3/envs/py38numba/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:39:20 2021

DNS Environment

@author: Alec Linot
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import math
import netCDF4
import sys
import os
from Solver import Solver
import pickle
import time
import numba


class DNS():
    
    def __init__(self,Vmax=1/20,T_act=5,T=99.9,T_save=0.2,savefield=False,Re=400,theta=0,actp=0,scale=1,obs_type='Full',obs_space=[cp.arange(6),cp.arange(3),cp.arange(2),cp.arange(3)]):
        self.theta=theta
        self.savefield=savefield
        self.T_act=T_act
        self.T=T
        self.T_save=T_save
        self.t=0
        self.dt=.02
        self.episode=0
        self.scale=scale

        # Initialize solver
        self.N=[32,35,32,3,1] # EDIT THIS, RUNS FASTEST FOR POWERS OF 2 ON X,Z, ODD NUMBERS FOR Y. ARRAY GPES X,Y,Z,Velocity(u,v,w),Number of stored fields
        self.x=[.875*2*math.pi,2,.3*2*math.pi] # TRY CHANGING .6 TO OTHER VALUES, e.g .3
        self.sol=Solver([self.N[1],self.N[0],self.N[2],self.N[3],self.N[4]],[self.x[1],self.x[0],self.x[2]],store=True,Re=Re)
        self.baseflow=self.sol.y
        
        # Initialize jets location
        self.phi=0
        self.slots=True

        # Set the top boundary condition to 0
        self.top=cp.zeros((self.N[0],self.N[2]),dtype=cp.complex128)

        # Actuator max settings
        self.j1=self.slot(math.pi/2,math.pi/2)
        self.j2=self.slot(math.pi/2,3*math.pi/2)
        F=self.BC([1,1])
        self.Fmax=cp.max(cp.real(cp.fft.ifftn(F)))
        self.Vmax=Vmax


        # RL stuff
        # Pick the observation type
        self.save=obs_space
        self.obs_type=obs_type
        if obs_type=='Spectral':
            self.observation_space = (len(obs_space[0])*len(obs_space[1])*len(obs_space[2])*len(obs_space[3])*2,) #TUPLE FORM TO MATCH BUFFER
        elif obs_type=='Physical':
            self.observation_space = (len(obs_space[0])*len(obs_space[1])*len(obs_space[2])*len(obs_space[3]),)
            
        # RL specific parameters
        self.actp=actp #actuation penalty
        self.action_space_high = 1.0
        self.action_space = 1   
        self.reward_range = [cp.NINF, 0]

        # Start on Initialization
        self.Out("{}\t\t{}\t\t\t{}\t\t\t{}\t{}".format('Time','Act','Energy','Reward([top,bot]=avg)','Comp Time'))

        # Initialize compute time
        self.compT=time.time()

    def Out(self,text,name='Out.txt'):
        print(text)
        newfile=open(name,'a+')
        newfile.write(text+'\n')
        newfile.close()

    def slot(self,phi1=0,phi2=0):
        scale=1*self.scale
        sig1=scale*self.sol.L[1]
        sig2=scale*self.sol.L[2]   
        temp=cp.exp(-(self.sol.k[0][0,:,:,0]**2/(2*sig1**2)+self.sol.k[1][0,:,:,0]**2/(2*sig2**2)))*cp.exp(-1j*(self.sol.k[0][0,:,:,0]*phi1+self.sol.k[1][0,:,:,0]*phi2))
        temp[1:,:]=0
        return temp

    def BC(self,a):
        F=a[0]*self.j1
        F+=a[1]*self.j2

        return F

    def step(self,act):

        # Set the actuation
        a=cp.asarray(act)
        # Concatenate the constrained action
        a=cp.concatenate((a,cp.asarray([-cp.sum(a)])))

        # Compute the top and bottom boundary conditions
        bot=self.BC(a)/self.Fmax*self.Vmax*1/self.sol.N[1]*1/self.sol.N[2]
        # Compute the new BC
        newBC=cp.concatenate((self.top[cp.newaxis,:,:],bot[cp.newaxis,:,:]),axis=0)

        self.Out('New action')
        done=False

        # Force u to be a single snapshot for new actuations, because the mutlistage method is used
        self.u=self.u[:,:,:,:,-1:]
        reward=[]
        # Loop through an action commitment time
        for i in range(int(self.T_act/self.dt)):

            # Time evolution
            if i<2: # Initial steps after new actuation
                uf,self.pf=self.sol.Step(self.u[:,:,:,:,-1:], self.pf[:,:,:,cp.newaxis,cp.newaxis], newBC,True,'Init',False,0,0,0,0)
                self.u=cp.concatenate((self.u,uf[:,:,:,:,cp.newaxis]),axis=-1)
            else: # multistep scheme
                if i<3: #This speeds up the scheme by saving 6 helmholtz solves (complimentary and A0 problems)
                    fixedBC=False
                else:
                    fixedBC=True
                    
                uf,self.pf=self.sol.Step(self.u,self.pf,newBC,True,'Multi',fixedBC,0,0,0,0)
                self.u=cp.concatenate((self.u[:,:,:,:,1:],uf[:,:,:,:,cp.newaxis]),axis=-1)

            # Output fields
            if i % int(self.T_save/self.dt)==0:
                # Save data
                if self.savefield==True:
                    
                    if os.path.isdir('./data')==False:
                        os.mkdir('./data')
                    epdir='./data/'+str(self.episode)
                    if os.path.isdir(epdir)==False:
                        os.mkdir(epdir)
                    # Save data
                    pickle.dump(cp.asnumpy(uf),open(epdir+'/u'+"{:.2f}".format(self.t)+'.p','wb'))
                    pickle.dump(cp.asnumpy(self.pf),open(epdir+'/q'+"{:.2f}".format(self.t)+'.p','wb'))
                    pickle.dump(cp.asnumpy(a),open(epdir+'/act'+"{:.2f}".format(self.t)+'.p','wb'))
                
                # Output compute time and other stats
                self.compT=time.time()-self.compT
                act_str=[item for sublist in [a.tolist()] for item in sublist]
                act_str=str([round(num, 1) for num in act_str])
                rew=self.sol.reward(uf,self.pf)
                rew_cpu = [float(x) for x in rew]
                avg = (rew_cpu[0] + rew_cpu[1]) / 2
                #rew_str=str([round(float(num),1) for num in rew_cpu])+'='+str(round(float((rew_cpu[0]+rew_cpu[1])/2),1))
                rew_str = f"{[round(num,1) for num in rew_cpu]}={round(avg,1)}"
                self.Out("{}\t\t{}\t{}\t\t{}\t\t{}".format("{:.2f}".format(self.t),act_str,"{:.4e}".format(cp.linalg.norm(uf)),rew_str,"{:.2f}".format(self.compT)))
                self.compT=time.time()
            
            # Increment time
            self.t+=self.dt
            # Compute the drag averaged over both walls
            rew=self.sol.reward(uf,self.pf)
            reward.append((rew[0]+rew[1])/2)   
            
            # Break if the code becomes unstable
            if cp.linalg.norm(uf)>10**2:
                done=True
                reward=-10**4
                break
        
        if self.t>self.T:
            done=True

        # Return the state, the reward, and a done condition
        reward=cp.mean(cp.asarray(reward))+self.actp*act**2
        u_save=self.obs(uf)

        return u_save, float(reward),done

    def obs(self,uf):

        if self.obs_type=='Spectral':
            u_save=cp.concatenate((cp.real(uf[self.save[0][:,cp.newaxis,cp.newaxis,cp.newaxis],self.save[1][:,cp.newaxis,cp.newaxis],self.save[2][:,cp.newaxis],self.save[3]]).flatten(),cp.imag(uf[self.save[0][:,cp.newaxis,cp.newaxis,cp.newaxis],self.save[1][:,cp.newaxis,cp.newaxis],self.save[2][:,cp.newaxis],self.save[3]]).flatten()))
            u_save=self.N[0]*self.N[2]*u_save
        elif self.obs_type=='Physical':
            # Convert to real
            uf=cp.real(self.sol.ifft(self.sol.icheb(uf[:,:,:,:,cp.newaxis])).squeeze())
            u_save=uf[self.save[0][:,cp.newaxis,cp.newaxis,cp.newaxis],self.save[1][:,cp.newaxis,cp.newaxis],self.save[2][:,cp.newaxis],self.save[3]].flatten()
        elif self.state_type=='Full':
            u_save=uf
            
        return u_save

    def reset(self,upath='../TestData/u'+str(cp.random.randint(25)+1)+'.p'):
        
        # Reset the time and actuation location
        self.t=0
        self.phi=0
        self.Out('New episode')
        self.episode+=1
        print(upath)
        us=pickle.load(open(upath,'rb'))
        us = cp.asarray(us)
        us=us[:,:,:,:,cp.newaxis]
        pf=0*us[:,:,:,0,0]
        
        # Add random phase, keep the current phase, or set a specific phase
        if self.theta=='Random':
            theta=cp.random.rand(1)*2*math.pi
            us=us*cp.exp(-1j*theta*self.sol.k[1])[:,:,:,:,cp.newaxis]
        elif self.theta=='None':
            pass
        else:
            theta=cp.arctan2(cp.imag(us[0,0,1,0,0]),cp.real(us[0,0,1,0,0]))+self.theta
            us=us*cp.exp(-1j*theta*self.sol.k[1])[:,:,:,:,cp.newaxis]

        # Keep the states for time evolution
        self.u=us
        self.pf=pf
        
        # Output the current observation
        uf=cp.squeeze(us)
        u_save=self.obs(uf)
        
        return u_save
    
def Time(text,name='Time.txt'):
    print(text)
    newfile=open(name,'a+')
    newfile.write(text+'\n')
    newfile.close()

#%% main function            
if __name__ == '__main__':
    cp.random.seed(0)
    # Load the environment (This outputs Fourier Chebyshev coefficients as the state. Set 'Spectral' to 'Full' to get the full state.)
    env=DNS(1/20,0.02,4.9,0.02,True,400,'None',1,1,'Spectral',[cp.arange(6),cp.arange(3),cp.arange(2),cp.arange(3)])
    
    # Loop over episodes
    n_episodes=1

    max_steps = 2 #Caps timing steps
    energies = []
    times = [] # Stores timings


    for ep in range(n_episodes):
        # Reset the environment
        observation = env.reset(upath="/scratch4/workspace/vkini_umass_edu-shared/Comparisons/data_numpy/1/u20.00.p")

        # Loop over actions
        done=False
        i=0
        while i < max_steps: #Caps loop
            # Pick an action
            act=0*(-cp.ones(1))
            start=time.time()
            
            # Run the environment for a step
            state,reward,done=env.step(act)

            end = time.time()
            
            #Records results
            energies.append(float(cp.linalg.norm(state)))
            times.append(end - start)


            # Output timing
            Time('Loop '+str(i))
            i+=1
            Time(str(time.time()-start))
            Time(str(state.shape))
    # Save reference data
    np.save(r"Results/New/energy_new.npy", cp.asnumpy(cp.array(energies)))
    np.save(r"Results/New/timing_new.npy", cp.asnumpy(cp.array(times)))
    print("Average step time:", cp.mean(cp.array(times)))


    cpu_pickle = pickle.load(open("/home/vkini_umass_edu/PyChannel/Slots/data/1/u0.02.p","rb"))
    # print(cpu_pickle.shape)
    gpu_pickle = pickle.load(open("/scratch4/workspace/vkini_umass_edu-shared/varun-cuda/PyChannel/Slots/data/1/u0.02.p","rb"))
    # print(gpu_pickle.shape)

    # diff = cpu_pickle - gpu_pickle

    # abs_err = np.linalg.norm(diff)
    # rel_err = abs_err / np.linalg.norm(cpu_pickle)

    # print("Absolute error:", abs_err)
    # print("Relative error:", rel_err)


    #Plot
    plt.figure()
    plt.plot(cp.asnumpy(cp.array(energies)), marker='o')
    plt.xlabel("Step")
    plt.ylabel("State (spectral energy proxy)")
    plt.title("Baseline short-run diagnostic")
    plt.grid(True)
    plt.show()    

    cp_cpu_pickle = cp.array(cpu_pickle)

    print("difference:", np.linalg.norm(cpu_pickle - cp.asnumpy(cp_cpu_pickle))) 

    u=env.sol.ifft(env.sol.icheb(cp.array(cpu_pickle)))
    u2=env.sol.ifft(env.sol.icheb(cp.array(gpu_pickle)))

    u_numpy = u.get()
    u2_numpy = u2.get()

    print(u_numpy.shape)
    print(u2_numpy.shape)
    print(np.linalg.norm(u-u2))
    plt.figure()
    plt.pcolormesh(u_numpy[4,:,:,0,0].real)
    plt.colorbar()
    plt.savefig("CPU.png")   

    plt.figure()
    plt.pcolormesh(u2_numpy[4,:,:,0,0].real)
    plt.colorbar()
    plt.savefig("GPU.png")          
