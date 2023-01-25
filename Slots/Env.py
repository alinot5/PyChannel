#!/home/linot/anaconda3/envs/py38numba/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:39:20 2021

DNS Environment

@author: Alec Linot
"""

import numpy as np
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
    
    def __init__(self,Vmax=1/20,T_act=5,T=99.9,T_save=0.2,savefield=False,Re=400,theta=0,actp=0,scale=1,obs_type='Full',obs_space=[np.arange(6),np.arange(3),np.arange(2),np.arange(3)]):
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
        self.N=[32,35,32,3,1]
        self.x=[.875*2*math.pi,2,.6*2*math.pi]
        self.sol=Solver([self.N[1],self.N[0],self.N[2],self.N[3],self.N[4]],[self.x[1],self.x[0],self.x[2]],store=True,Re=Re)
        self.baseflow=self.sol.y
        
        # Initialize jets location
        self.phi=0
        self.slots=True

        # Set the top boundary condition to 0
        self.top=np.zeros((self.N[0],self.N[2]),dtype='complex128')

        # Actuator max settings
        self.j1=self.slot(math.pi/2,math.pi/2)
        self.j2=self.slot(math.pi/2,3*math.pi/2)
        F=self.BC([1,1])
        self.Fmax=np.max(np.real(np.fft.ifftn(F)))
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
        self.reward_range = [np.NINF, 0]

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
        temp=np.exp(-(self.sol.k[0][0,:,:,0]**2/(2*sig1**2)+self.sol.k[1][0,:,:,0]**2/(2*sig2**2)))*np.exp(-1j*(self.sol.k[0][0,:,:,0]*phi1+self.sol.k[1][0,:,:,0]*phi2))
        temp[1:,:]=0
        return temp

    def BC(self,a):
        F=a[0]*self.j1
        F+=a[1]*self.j2

        return F

    def step(self,act):

        # Set the actuation
        a=np.asarray(act)
        # Concatenate the constrained action
        a=np.concatenate((a,np.asarray([-np.sum(a)])))

        # Compute the top and bottom boundary conditions
        bot=self.BC(a)/self.Fmax*self.Vmax*1/self.sol.N[1]*1/self.sol.N[2]
        # Compute the new BC
        newBC=np.concatenate((self.top[np.newaxis,:,:],bot[np.newaxis,:,:]),axis=0)

        self.Out('New action')
        done=False

        # Force u to be a single snapshot for new actuations, because the mutlistage method is used
        self.u=self.u[:,:,:,:,-1:]
        reward=[]
        # Loop through an action commitment time
        for i in range(int(self.T_act/self.dt)):

            # Time evolution
            if i<2: # Initial steps after new actuation
                uf,self.pf=self.sol.Step(self.u[:,:,:,:,-1:],self.pf[:,:,:,np.newaxis,np.newaxis],newBC,True,'Init',False,0,0,0,0)
                self.u=np.concatenate((self.u,uf[:,:,:,:,np.newaxis]),axis=-1)
            else: # multistep scheme
                if i<3: #This speeds up the scheme by saving 6 helmholtz solves (complimentary and A0 problems)
                    fixedBC=False
                else:
                    fixedBC=True
                    
                uf,self.pf=self.sol.Step(self.u,self.pf,newBC,True,'Multi',fixedBC,0,0,0,0)
                self.u=np.concatenate((self.u[:,:,:,:,1:],uf[:,:,:,:,np.newaxis]),axis=-1)

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
                    pickle.dump(uf,open(epdir+'/u'+"{:.2f}".format(self.t)+'.p','wb'))
                    pickle.dump(self.pf,open(epdir+'/q'+"{:.2f}".format(self.t)+'.p','wb'))
                    pickle.dump(a,open(epdir+'/act'+"{:.2f}".format(self.t)+'.p','wb'))
                
                # Output compute time and other stats
                self.compT=time.time()-self.compT
                act_str=[item for sublist in [a.tolist()] for item in sublist]
                act_str=str([round(num, 1) for num in act_str])
                rew=self.sol.reward(uf,self.pf)
                rew_str=str([round(num,1) for num in rew])+'='+str(round((rew[0]+rew[1])/2,1))
                self.Out("{}\t\t{}\t{}\t\t{}\t\t{}".format("{:.2f}".format(self.t),act_str,"{:.4e}".format(np.linalg.norm(uf)),rew_str,"{:.2f}".format(self.compT)))
                self.compT=time.time()
            
            # Increment time
            self.t+=self.dt
            # Compute the drag averaged over both walls
            rew=self.sol.reward(uf,self.pf)
            reward.append((rew[0]+rew[1])/2)   
            
            # Break if the code becomes unstable
            if np.linalg.norm(uf)>10**2:
                done=True
                reward=-10**4
                break
        
        if self.t>self.T:
            done=True

        # Return the state, the reward, and a done condition
        reward=np.mean(np.asarray(reward))+self.actp*act**2
        u_save=self.obs(uf)

        return u_save,reward,done

    def obs(self,uf):

        if self.obs_type=='Spectral':
            u_save=np.concatenate((np.real(uf[self.save[0][:,np.newaxis,np.newaxis,np.newaxis],self.save[1][:,np.newaxis,np.newaxis],self.save[2][:,np.newaxis],self.save[3]]).flatten(),np.imag(uf[self.save[0][:,np.newaxis,np.newaxis,np.newaxis],self.save[1][:,np.newaxis,np.newaxis],self.save[2][:,np.newaxis],self.save[3]]).flatten()))
            u_save=self.N[0]*self.N[2]*u_save
        elif self.obs_type=='Physical':
            # Convert to real
            uf=np.real(self.sol.ifft(self.sol.icheb(uf[:,:,:,:,np.newaxis])).squeeze())
            u_save=uf[self.save[0][:,np.newaxis,np.newaxis,np.newaxis],self.save[1][:,np.newaxis,np.newaxis],self.save[2][:,np.newaxis],self.save[3]].flatten()
        elif self.state_type=='Full':
            u_save=uf
            
        return u_save

    def reset(self,upath='../TestData/u'+str(np.random.randint(25))+'.p'):
        
        # Reset the time and actuation location
        self.t=0
        self.phi=0
        self.Out('New episode')
        self.episode+=1
        
        us=pickle.load(open(upath,'rb'))
        us=us[:,:,:,:,np.newaxis]
        pf=0*us[:,:,:,0,0]

        # Add random phase, keep the current phase, or set a specific phase
        if self.theta=='Random':
            theta=np.random.rand(1)*2*math.pi
            us=us*np.exp(-1j*theta*self.sol.k[1])[:,:,:,:,np.newaxis]
        elif self.theta=='None':
            pass
        else:
            theta=np.arctan2(np.imag(us[0,0,1,0,0]),np.real(us[0,0,1,0,0]))+self.theta
            us=us*np.exp(-1j*theta*self.sol.k[1])[:,:,:,:,np.newaxis]

        # Keep the states for time evolution
        self.u=us
        self.pf=pf
        
        # Output the current observation
        uf=np.squeeze(us)
        u_save=self.obs(uf)
        
        return u_save
    
def Time(text,name='Time.txt'):
    print(text)
    newfile=open(name,'a+')
    newfile.write(text+'\n')
    newfile.close()

#%% main function            
if __name__ == '__main__':
    
    # Load the environment (This outputs Fourier Chebyshev coefficients as the state. Set 'Spectral' to 'Full' to get the full state.)
    env=DNS(1/20,5,19.9,1,True,400,'None',1,1,'Spectral',[np.arange(6),np.arange(3),np.arange(2),np.arange(3)])
    
    # Loop over episodes
    n_episodes=1
    for ep in range(n_episodes):
        # Reset the environment
        observation = env.reset()
        
        # Loop over actions
        done=False
        i=0
        while not done:
            # Pick an action
            act=0*(-np.ones(1))
            start=time.time()
            
            # Run the environment for a step
            state,reward,done=env.step(act)
            
            # Output timing
            Time('Loop '+str(i))
            i+=1
            Time(str(time.time()-start))
            Time(str(state.shape))


