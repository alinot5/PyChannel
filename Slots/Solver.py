#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:39:20 2021

Oscillating wall code

@author: Alec
"""

from numba import jit
from numba import set_num_threads
from numba import njit, prange
from numba import int64,float64,complex128
import numpy as np
import matplotlib.pyplot as plt
import math
import netCDF4
import sys
from Derivatives import Derivatives
import time
import os
import scipy
# u is organized y,x,z

class Solver(Derivatives):
    
    def __init__(self,N=[35,32,32,3,1],L=[2,.875*2*math.pi,.6*2*math.pi],store=True,Re=400):
        super(Solver,self).__init__(N,L,Re)
        self.flag=0
        self.N=N#[Ny,Nx,Nz]
        self.L=L
        # Get the y grid and baseflow
        y=np.arange(N[0])
        self.y=-np.cos(y*math.pi/(N[0]-1))
        
        self.base=np.zeros((N[0],N[1],N[2],N[3]))
        self.base[:,:,:,0]=np.flip(np.repeat(np.repeat(self.y[:,np.newaxis,np.newaxis],N[1],axis=1),N[2],axis=2),0)

        # Get the wavenumbers
        k=np.meshgrid(np.fft.fftfreq(N[1], d=1/N[1]),np.fft.fftfreq(N[2], d=1/N[2]))
        k=[k[0].transpose(),k[1].transpose()]
        k=[np.repeat(k[0][np.newaxis,:,:],N[0],axis=0),np.repeat(k[1][np.newaxis,:,:],N[0],axis=0)]
        self.k=[np.repeat(k[0][:,:,:,np.newaxis],N[-2],axis=-1).astype('complex128'),np.repeat(k[1][:,:,:,np.newaxis],N[-2],axis=-1).astype('complex128')]
        

        # Parameters
        self.Re=Re
        self.dt=.02
        self.k2=4*math.pi**2*(self.k[0][0,:,:,0]**2/L[1]**2+self.k[1][0,:,:,0]**2/L[2]**2)

        # Save lambda and constants each scheme
        # Initialization scheme (SMRK2)
        self.alpha=[29/96,-3/40,1/6]
        self.beta=[37/160,5/24,1/6]
        self.gamma=[8/15,5/12,3/4]
        self.zeta=[0,-17/60,-5/12]
        self.lam_init=[1/(self.dt*self.beta[0])+self.k2/self.Re,1/(self.dt*self.beta[1])+self.k2/self.Re,1/(self.dt*self.beta[2])+self.k2/self.Re]

        # Multistep scheme (SBDF3)
        self.a=[11/6,-3,3/2,-1/3]
        self.b=[3,-3,1]
        self.lam=self.a[0]/self.dt+self.k2/self.Re

        # Store vectors for P,Q and R (used in Helmholtz solver)
        self.Ps=np.zeros((self.N[0]-2))
        self.Qs=np.zeros((self.N[0]-2))
        self.Rs=np.zeros((self.N[0]-2))
        for i in range(2,self.N[0]):
            self.Ps[i-2]=self.P(i)
            self.Qs[i-2]=self.Q(i)
            self.Rs[i-2]=self.R(i)

        # RHS of the A0 problem
        self.A0BC=np.repeat(np.repeat([[[0]],[[0]]],self.N[2],axis=-1),self.N[1],axis=-2).astype('complex128') # Is this right?
        self.ck=np.ones((self.N[0],self.N[1],self.N[2]),dtype=np.complex128); self.ck[0,:,:]=.5; self.ck[-1,:,:]=0; self.ck=self.ck/self.Re 
        # Matrices used multiple times
        self.nones=(-np.ones(self.N[0]))**np.arange(self.N[0])
        [self.Mfe,self.Mfo]=self.Mf()

        # Initialize things for the Helmholtz solver
        self.utemp=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        self.Ne=int((self.N[0]-1)/2+1)
        self.No=self.N[0]-self.Ne
        self.Me1=np.ones((1,self.Ne),dtype=np.complex128)
        self.Mo1=np.ones((1,self.No),dtype=np.complex128)
        self.Ie=np.eye(self.Ne-1,dtype=np.complex128)
        self.Io=np.eye(self.No-1,dtype=np.complex128)

        # Store the necessary inverse matrices for the helmholtz solver 
        # this speeds up the code a good bit, but should be removed if storage costs are an issue
        # also the computation may be faster to do np.linag.solve if it can be computed with a sparse matrix solver
        self.style='0'
        if store:
            # pressure calcs
            lam=self.k2
            self.MeInvP=np.zeros((self.Ne,self.Ne,self.N[1],self.N[2]),dtype=np.complex128)
            self.MoInvP=np.zeros((self.No,self.No,self.N[1],self.N[2]),dtype=np.complex128)
            for i in range(self.N[1]):
                for j in range(self.N[2]):
                    Me=np.concatenate((self.Me1,-lam[i,j]*self.Mfe[1:,:]),axis=0)
                    Me[1:,1:]=Me[1:,1:]+self.Ie
                    Mo=np.concatenate((self.Mo1,-lam[i,j]*self.Mfo[1:,:]),axis=0)
                    Mo[1:,1:]=Mo[1:,1:]+self.Io

                    # compute inverse
                    self.MeInvP[:,:,i,j]=np.linalg.inv(Me)
                    self.MoInvP[:,:,i,j]=np.linalg.inv(Mo)

            # velocity calcs
            lam=self.lam*self.Re
            self.MeInvu=np.zeros((self.Ne,self.Ne,self.N[1],self.N[2]),dtype=np.complex128)
            self.MoInvu=np.zeros((self.No,self.No,self.N[1],self.N[2]),dtype=np.complex128)
            for i in range(self.N[1]):
                for j in range(self.N[2]):
                    Me=np.concatenate((self.Me1,-lam[i,j]*self.Mfe[1:,:]),axis=0)
                    Me[1:,1:]=Me[1:,1:]+self.Ie
                    Mo=np.concatenate((self.Mo1,-lam[i,j]*self.Mfo[1:,:]),axis=0)
                    Mo[1:,1:]=Mo[1:,1:]+self.Io

                    # compute inverse
                    self.MeInvu[:,:,i,j]=np.linalg.inv(Me)
                    self.MoInvu[:,:,i,j]=np.linalg.inv(Mo)

        # Store Nonlinear RHS terms
        self.H2=[0]
        self.H3=[0]

        # Store compute times
        self.homt=0
        self.St=0
        self.A0t=0
        self.A1t=0
        self.taut=0
        self.uwt=0

        # Setup numba
        #threads=len(os.sched_getaffinity(0))
        threads=1
        set_num_threads(threads)


    # The fastest way to do this is using numba! Now this calculation is 10 times faster than the S calculation
    def Helmholtz_Loop(self,lam,f,BC,style='solve'):
        if style=='P':
            temp=self.Helmholtz_Loop_Inv(self.N[1],self.N[2],self.Mfe[1:,:],self.Mfo[1:,:],f[::2,:,:],f[1:-1:2,:,:],BC,self.utemp,self.MeInvP,self.MoInvP)
        elif style=='u':
            temp=self.Helmholtz_Loop_Inv(self.N[1],self.N[2],self.Mfe[1:,:],self.Mfo[1:,:],f[::2,:,:],f[1:-1:2,:,:],BC,self.utemp,self.MeInvu,self.MoInvu)
        else:
            temp=self.Helmholtz_Loop_par_pass(self.N[1],self.N[2],lam,self.Mfe[1:,:],self.Mfo[1:,:],f[::2,:,:],f[1:-1:2,:,:],BC,self.utemp,self.Me1,self.Mo1,self.Ie,self.Io)

        return temp
    
    @staticmethod
    @jit(complex128[:,:,:](int64, int64,complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:]),parallel=True)
    def Helmholtz_Loop_par_pass(N1,N2,beta,Mfe,Mfo,fe,fo,g,u,Me1,Mo1,Ie,Io):
        for i in prange(N1):
            for j in prange(N2):  
                # This approach appears to be marginally faster than for loops
                Me=np.concatenate((Me1,-beta[i,j]*Mfe),axis=0)
                Me[1:,1:]=Me[1:,1:]+Ie
                Mo=np.concatenate((Mo1,-beta[i,j]*Mfo),axis=0)
                Mo[1:,1:]=Mo[1:,1:]+Io

                # Get the BC
                ge=(g[0,i,j:j+1]+g[1,i,j:j+1])/2
                go=(g[0,i,j:j+1]-g[1,i,j:j+1])/2

                # Solve the two problems for ue and uo
                Fe=np.concatenate((ge,Mfe@fe[:,i,j]))
                u[::2,i,j]=np.linalg.solve(Me,Fe)
                Fo=np.concatenate((go,Mfo@fo[:,i,j]))
                u[1::2,i,j]=np.linalg.solve(Mo,Fo)

        return u

    @staticmethod
    @jit(complex128[:,:,:](int64, int64,complex128[:,:],complex128[:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:,:],complex128[:,:,:,:]),parallel=True)
    def Helmholtz_Loop_Inv(N1,N2,Mfe,Mfo,fe,fo,g,u,MeI,MoI):
        for i in prange(N1):
            for j in prange(N2):  
                # Get the BC
                ge=(g[0,i,j:j+1]+g[1,i,j:j+1])/2
                go=(g[0,i,j:j+1]-g[1,i,j:j+1])/2

                # Solve the two problems for ue and uo
                Fe=np.concatenate((ge,Mfe@fe[:,i,j]))
                u[::2,i,j]=MeI[:,:,i,j]@Fe
                Fo=np.concatenate((go,Mfo@fo[:,i,j]))
                u[1::2,i,j]=MoI[:,:,i,j]@Fo

        return u

    def P(self,k):
        n=k-2
        c=1
        if n==0:
            c=2
        return c/(4*k*(k-1))
    
    def Q(self,k):
        n=k
        e=1
        if n>self.N[0]-3:
            e=0
        return -e/(2*(k**2-1))

    def R(self,k):
        n=k+2
        e=1
        if n>self.N[0]-3:
            e=0
        # if n>self.N[0]-2:
        #     e=0
        return e/(4*k*(k+1))
        
    # I only have to get Mf once
    def Mf(self,):
        Ne=int((self.N[0]-1)/2+1)
        No=self.N[0]-Ne
        Me=np.zeros((Ne,Ne),dtype=np.complex128)
        Mo=np.zeros((No,No),dtype=np.complex128)
        
        # Loop through the elements in the matrix
        for i in range(2,self.N[0],2):
            idx=int(i/2)
            #Me
            Me[idx,idx-1]=self.Ps[i-2]
            Me[idx,idx]=self.Qs[i-2]
            if i!=self.N[0]-1:
                Me[idx,idx+1]=self.Rs[i-2]
            
            #Mo
            j=i+1
            if idx+1<=No:
                Mo[idx,idx-1]=self.Ps[j-2]
                Mo[idx,idx]=self.Qs[j-2]
                if j!=self.N[0]-2:
                    Mo[idx,idx+1]=self.Rs[j-2]
                
        return Me,Mo
   
    def compl_store(self,BC,lam):
        #Compute the complementary solutions for P and u2
        P1c=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        v1c=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        P2c=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        v2c=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)

        # Compute the pressure homogenous solutions
        PBC=np.repeat(np.repeat([[[1]],[[0]]],self.N[2],axis=-1),self.N[1],axis=-2).astype('complex128') # Is this right?
        P1c=np.copy(self.Helmholtz_Loop(self.k2,np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128),PBC,'P'+self.style))
        PBC=np.repeat(np.repeat([[[0]],[[1]]],self.N[2],axis=-1),self.N[1],axis=-2).astype('complex128') # Is this right?
        P2c=np.copy(self.Helmholtz_Loop(self.k2,np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128),PBC,'P'+self.style))

        # Compute RHS for helmholtz solver
        fv1=self.dy(P1c[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze()
        fv2=self.dy(P2c[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze()
 
        # Compute the velocity homogenous solutions #For some reason running this Helmholtz loop was changing the values of P1c and P2c
        v1c=np.copy(self.Helmholtz_Loop(lam*self.Re,fv1*self.Re,BC,'u'+self.style))
        v2c=np.copy(self.Helmholtz_Loop(lam*self.Re,fv2*self.Re,BC,'u'+self.style))

        # Set mean to 0 to enforce
        P1c[:,0,0]=0
        P2c[:,0,0]=0
        v1c[:,0,0]=0
        v2c[:,0,0]=0

        #Compute c1 and c2
        Dv1=self.dy(v1c[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze()
        Dv2=self.dy(v2c[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze()
        #Make the matrix
        Dv1t=np.sum(Dv1,axis=0)
        Dv1b=np.sum(Dv1*self.nones[:,np.newaxis,np.newaxis],axis=0)
        Dv2t=np.sum(Dv2,axis=0)
        Dv2b=np.sum(Dv2*self.nones[:,np.newaxis,np.newaxis],axis=0)
        Dv=np.asarray([[Dv1t,Dv2t],[Dv1b,Dv2b]]).squeeze()
        # Setting this to prevent nan
        Dv[:,:,0,0]=np.eye(2)

        return Dv,P1c,P2c,v1c,v2c

    def S_SBDF3(self,u,inpy=0,outy=0,inpxz=0,outxz=0):
        if inpxz==1 and outxz==0:
            u=self.fft(u)
        if inpxz==0 and outxz==1:
            u=self.ifft(u)

        if inpy==1 and outy==0:
            u=self.cheb(u)
        if inpy==0 and outy==1:
            u=self.icheb(u)

        Lin=1/self.dt*(self.a[1]*u[:,:,:,:,2:3]+self.a[2]*u[:,:,:,:,1:2]+self.a[3]*u[:,:,:,:,0:1])
        if isinstance(self.H2,list):
            self.H2=self.H(u[:,:,:,:,1:2],inpy,outy,inpxz,outxz)
        if isinstance(self.H3,list):
            self.H3=self.H(u[:,:,:,:,0:1],inpy,outy,inpxz,outxz)
        H1=self.H(u[:,:,:,:,2:3],inpy,outy,inpxz,outxz)
        Nonlin=self.b[0]*H1+self.b[1]*self.H2+self.b[2]*self.H3
        S=Lin-Nonlin
        Sp=-self.div(S,inpy,outy,inpxz,outxz)
        
        # Store nonlinearity
        self.H3=self.H2
        self.H2=H1
        return S.squeeze(),Sp.squeeze()

    def S_SMRK2(self,u,P,i,inpy=0,outy=0,inpxz=0,outxz=0):
        if inpxz==1 and outxz==0:
            u=self.fft(u)
            p=self.fft(p)
        if inpxz==0 and outxz==1:
            u=self.ifft(u)
            p=self.ifft(p)

        if inpy==1 and outy==0:
            u=self.cheb(u)
            p=self.cheb(p)
        if inpy==0 and outy==1:
            u=self.icheb(u)
            p=self.icheb(p)

        gradp=self.alpha[i]/self.beta[i]*self.gradu(P,inpy,outy,inpxz,outxz)[:,:,:,:,:,0]
        Lin=-1/(self.dt*self.beta[i])*u[:,:,:,:,i:]-self.alpha[i]/(self.Re*self.beta[i])*self.lap(u[:,:,:,:,i:],inpy,outy,inpxz,outxz)
        Lin=Lin-gradp
        if i==0:
            Nonlin=-self.gamma[i]/self.beta[i]*self.H(u[:,:,:,:,i:],inpy,outy,inpxz,outxz)
        else:
            Nonlin=-self.gamma[i]/self.beta[i]*self.H(u[:,:,:,:,i:],inpy,outy,inpxz,outxz)-self.zeta[i]/self.beta[i]*self.H(u[:,:,:,:,i-1:i],inpy,outy,inpxz,outxz)
        
        S=Lin+Nonlin
        Sp=-self.div(S,inpy,outy,inpxz,outxz)

        return S.squeeze(),Sp.squeeze()

    def Pv(self,S,Sp,BC,lam,inpy=0,outy=0,inpxz=0,outxz=0): # Solves P''-k2P=Sp 1/Rev''-lam v- P'=S
        # convert all fields to chebyshev fourier
        if inpy==1:
            S=self.cheb(S[:,:,:,np.newaxis,np.newaxis]).squeeze()
            Sp=self.cheb(Sp[:,:,:,np.newaxis,np.newaxis]).squeeze()
        if inpxz==1:
            S=self.fft(S[:,:,:,np.newaxis,np.newaxis]).squeeze()
            Sp=self.fft(Sp[:,:,:,np.newaxis,np.newaxis]).squeeze()

        #######################################################################
        # Compute the particular solutions for P and u2 and the cs from the complimentary sol
        #######################################################################       
        Pp=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        vp=np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128)
        c=np.zeros((2,self.N[1],self.N[2]),dtype=np.complex128)
        ctemp=np.zeros((2,self.N[1],self.N[2]),dtype=np.complex128)
        # Get the particular solution for the pressure
        PBC=np.repeat(np.repeat([[[0]],[[0]]],self.N[2],axis=-1),self.N[1],axis=-2).astype('complex128') # Is this right?
        Pp=np.copy(self.Helmholtz_Loop(self.k2,Sp,PBC,'P'+self.style))

        # Compute the RHS used in the Helmholtz solver
        fv=S+self.dy(Pp[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze() # Solving this outside the loop is much faster
        # Get the particular solution for the velocity and solver for the pressure BC
        vp=np.copy(self.Helmholtz_Loop(lam*self.Re,fv*self.Re,BC,'u'+self.style))  

        # Set mean to 0 to enforce
        Pp[:,0,0]=0
        vp[:,0,0]=0

        # Solve for the pressure boundary conditions
        Dvp=self.dy(vp[:,:,:,np.newaxis,np.newaxis],0,0,0,0).squeeze()
        Dvpt=np.sum(Dvp,axis=0)
        Dvpb=np.sum(Dvp*self.nones[:,np.newaxis,np.newaxis],axis=0)
        b=np.asarray([[Dvpt],[Dvpb]]).squeeze()
        c=[self.Dv[1,1,:,:]*-b[0,:,:]-self.Dv[0,1,:,:]*-b[1,:,:] ,-self.Dv[1,0,:,:]*-b[0,:,:]+self.Dv[0,0,:,:]*-b[1,:,:]]/(self.Dv[0,0,:,:]*self.Dv[1,1,:,:]-self.Dv[0,1,:,:]*self.Dv[1,0,:,:])
        c[:,0,0]=np.zeros(2)

        # This is equivalent (it may not have been with the original pressure)
        P=Pp+c[0:1,:,:]*self.P1c+c[1:,:,:]*self.P2c
        v=vp+c[0:1,:,:]*self.v1c+c[1:,:,:]*self.v2c

        if outy==1:
            P=self.icheb(P[:,:,:,np.newaxis,np.newaxis]).squeeze()
            v=self.icheb(v[:,:,:,np.newaxis,np.newaxis]).squeeze()
        if outxz==1:
            P=self.ifft(P[:,:,:,np.newaxis,np.newaxis]).squeeze()
            v=self.ifft(v[:,:,:,np.newaxis,np.newaxis]).squeeze()

        return P,v

    def uw(self,S,P,v,lam,inpy=0,outy=0,inpxz=0,outxz=0):
        # convert all fields to chebyshev fourier
        if inpy==1:
            S=self.cheb(S[:,:,:,np.newaxis,np.newaxis]).squeeze()
            P=self.cheb(P[:,:,:,np.newaxis,np.newaxis]).squeeze()
            v=self.cheb(v[:,:,:,np.newaxis,np.newaxis]).squeeze()
        if inpxz==1:
            S=self.fft(S[:,:,:,np.newaxis,np.newaxis]).squeeze()
            P=self.fft(P[:,:,:,np.newaxis,np.newaxis]).squeeze()
            v=self.fft(v[:,:,:,np.newaxis,np.newaxis]).squeeze()

        # Compute the RHS used in the Helmholtz solver
        # this is the RHS for u
        fu=S[:,:,:,0]+1j*self.k[0][0,:,:,0]*2*math.pi/self.L[1]*P
        # this is the RHS for w
        fw=S[:,:,:,2]+1j*self.k[1][0,:,:,0]*2*math.pi/self.L[2]*P

        # Solve for u and w
        BC=np.repeat(np.repeat([[[0]],[[0]]],self.N[2],axis=-1),self.N[1],axis=-2).astype('complex128') # Is this right?
        w=np.copy(self.Helmholtz_Loop(lam*self.Re,fw*self.Re,BC,'u'+self.style))
        BC[:,0,0]=[1,-1]
        u=np.copy(self.Helmholtz_Loop(lam*self.Re,fu*self.Re,BC,'u'+self.style))
        u=np.concatenate((u[:,:,:,np.newaxis],v[:,:,:,np.newaxis],w[:,:,:,np.newaxis]),axis=-1)

        # Zero out values that should remain 0 for stability
        if self.N[1]%2==0:
            u[:,int(self.N[1]/2),:,:]=0
        if self.N[2]%2==0:
            u[:,:,int(self.N[2]/2),:]=0
        u[:,0,0,1]=0
        u[-1,1:,0,0]=0
        u[-1,0,1:,2]=0
        u[-1,1:,0,0]=np.real(u[-1,1:,0,0])
        u[-1,0,1:,2]=np.real(u[-1,0,1:,2])
        u[:,0,0,:]=np.real(u[:,0,0,:])

        if outy==1:
            u=self.icheb(u[:,:,:,:,np.newaxis]).squeeze()
        if outxz==1:
            u=self.ifft(u[:,:,:,:,np.newaxis]).squeeze()
            
        return u  

    def tau_cor(self,P0,v0,P1,v1,S,lam):
        # Constants
        #beta=self.Re/self.dt+self.k2
        beta=lam*self.Re
        N2=self.N[0]-1

        # Compute sigma for The A0 problem
        sig00=2*(N2-1)*(beta*v0[-2:-1,:,:]+2*N2*P0[-1:,:,:]*self.Re)
        sig01=2*N2*beta*v0[-1:,:,:]
        sig0=np.concatenate((sig00,sig01),axis=0)

        # Compute sigma for The A1 problem
        sig10=2*(N2-1)*(self.Re*S[-2:-1,:,:]+beta*v1[-2:-1,:,:]+2*N2*P1[-1:,:,:]*self.Re)
        sig11=2*N2*(self.Re*S[-1:,:,:]+beta*v1[-1:,:,:])
        sig1=np.concatenate((sig10,sig11),axis=0)

        # Compute the correction
        sigt=sig1/(1-sig0)
        idx=int(np.ceil(self.N[0]/2))
        sigt=np.repeat(sigt,idx,axis=0)
        sigp=np.zeros(sigt.shape,dtype=np.complex128)
        sigp[0::2,:,:]=sigt[:idx,:,:]
        sigp[1::2,:,:]=sigt[idx:,:,:]
        sigv=np.roll(sigp,1,axis=0)
        sigp=sigp[:-1,:,:]
        sigv=sigv[:-1,:,:]     

        # Correct the pressure and velocity fields
        P=P1+sigp*P0
        v=v1+sigv*v0
        v[:,0,0]=0
        return P,v

    def reward(self,u,p):
        # Compute the wall stress
        tau=np.real(self.dy(u[:,:,:,0:1,np.newaxis],0,1,0,1).squeeze())
        taut=tau[-1,:,:]
        taub=tau[0,:,:]

        # Compute the drag on the wall
        dA=self.L[1]/self.N[1]*self.L[2]/self.N[2]
        dragt=np.sum(taut-1)*dA
        dragb=np.sum(taub-1)*dA

        r=[-dragt,-dragb]
        return r

    def Step(self,u,P,BC,tau=True,scheme='Multi',fixedBC=False,inpy=0,outy=0,inpxz=0,outxz=0):

        if scheme=='Multi':
            self.style=''
            # Solve for the homogenous solutions
            temp=time.time()
            if fixedBC==False:
                self.Dv,self.P1c,self.P2c,self.v1c,self.v2c=self.compl_store(0*BC,self.lam)
                self.P0,self.v0=self.Pv(np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128),self.ck,self.A0BC,self.lam,0,0,0,0) # depends on the comp sol
            
            self.homt+=time.time()-temp
            # Compute S and Sp
            temp=time.time()
            S,Sp=self.S_SBDF3(u,inpy,0,inpxz,0)
            self.St+=time.time()-temp
            # A1 problem
            temp=time.time()
            P1,v1=self.Pv(S[:,:,:,1],Sp,BC,self.lam,0,0,0,0)
            self.A1t+=time.time()-temp
            # Tau correction
            if tau==True:
                # Compute the A0 problem
                temp=time.time()
                self.style=''
                self.A0t+=time.time()-temp
                # Use the A0 and A1 problems to solve for P and v
                temp=time.time()
                P,v=self.tau_cor(self.P0,self.v0,P1,v1,S[:,:,:,1],self.lam)
                self.taut+=time.time()-temp
            else:
                P=P1
                v=v1
            # Compute the new velocity field
            temp=time.time()
            unew=self.uw(S,P,v,self.lam,0,outy,0,outxz)
            self.uwt+=time.time()-temp

        else:
            self.style='0'
            # Loop overstages
            utemp=np.copy(u)
            Ptemp=np.copy(P)
            for i in range(3):
                # Solve for the homogenous solutions
                self.Dv,self.P1c,self.P2c,self.v1c,self.v2c=self.compl_store(0*BC,self.lam_init[i])
                # Compute S and Sp
                S,Sp=self.S_SMRK2(utemp,Ptemp,i,inpy,0,inpxz,0)
                # A1 problem
                P1,v1=self.Pv(S[:,:,:,1],Sp,BC,self.lam_init[i],0,0,0,0)
                # Tau correction
                if tau==True:
                    # Compute the A0 problem
                    P0,v0=self.Pv(np.zeros((self.N[0],self.N[1],self.N[2]),dtype=np.complex128),self.ck,self.A0BC,self.lam_init[i],0,0,0,0)
                    # Use the A0 and A1 problems to solve for P and v
                    P,v=self.tau_cor(P0,v0,P1,v1,S[:,:,:,1],self.lam_init[i])
                else:
                    P=P1
                    v=v1
                # Compute the new velocity field
                unew=self.uw(S,P,v,self.lam_init[i],0,outy,0,outxz)

                utemp=np.concatenate((utemp,unew[:,:,:,:,np.newaxis]),axis=-1)
                Ptemp=P[:,:,:,np.newaxis,np.newaxis]

        return unew,P