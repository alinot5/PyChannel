#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:39:20 2021

Oscillating wall code

@author: Alec
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import netCDF4
import sys
import time
import pyfftw
import os
# u is organized y,x,z

class Derivatives:
    
    def __init__(self,N=[35,32,32,3,1],L=[2,.875*2*math.pi,.6*2*math.pi],Re=400):
        self.N=N#[Ny,Nx,Nz]
        self.L=L
        
        k=np.meshgrid(np.fft.fftfreq(N[1], d=1/N[1]),np.fft.fftfreq(N[2], d=1/N[2]))
        k=[k[0].transpose(),k[1].transpose()]
        k=[np.repeat(k[0][np.newaxis,:,:],N[0],axis=0),np.repeat(k[1][np.newaxis,:,:],N[0],axis=0)]
        self.k=[np.repeat(k[0][:,:,:,np.newaxis],N[-2],axis=-1),np.repeat(k[1][:,:,:,np.newaxis],N[-2],axis=-1)]

        y=np.arange(N[0])
        self.y=-np.cos(y*math.pi/(N[0]-1))
        self.Re=Re
        
        # Store FFT things
        #threads=len(os.sched_getaffinity(0))
        threads=1
        # velocity
        ufftwb = pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],self.N[3],1), dtype='float64')
        ufftwf= pyfftw.empty_aligned((self.N[0],self.N[1], int(self.N[2]/2)+1,self.N[3],1), dtype='complex128')
        self.ufft_temp=np.zeros((self.N[0],self.N[1], self.N[2],self.N[3],1),dtype='complex128')
        self.ufft_object = pyfftw.FFTW(ufftwb, ufftwf,axes=(1,2),flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=threads)
        # pressure
        Pfftwb = pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],1,1), dtype='float64')
        Pfftwf= pyfftw.empty_aligned((self.N[0],self.N[1], int(self.N[2]/2)+1,1,1), dtype='complex128')
        self.Pfft_temp=np.zeros((self.N[0],self.N[1], self.N[2],1,1),dtype='complex128')
        self.Pfft_object = pyfftw.FFTW(Pfftwb, Pfftwf,axes=(1,2),flags=('FFTW_MEASURE',), direction='FFTW_FORWARD',threads=threads)
        # Store IFFT things
        # velocity
        uifftwb = pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],self.N[3],1), dtype='float64')
        uifftwf= pyfftw.empty_aligned((self.N[0],self.N[1], int(self.N[2]/2)+1,self.N[3],1), dtype='complex128')
        self.uifft_object = pyfftw.FFTW(uifftwf, uifftwb,axes=(1,2),flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD',threads=threads)
        # pressure
        Pifftwb = pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],1,1), dtype='float64')
        Pifftwf= pyfftw.empty_aligned((self.N[0],self.N[1], int(self.N[2]/2)+1,1,1), dtype='complex128')
        self.Pifft_object = pyfftw.FFTW(Pifftwf, Pifftwb,axes=(1,2),flags=('FFTW_MEASURE',), direction='FFTW_BACKWARD',threads=threads)

        # Store Cheb things
        #velocity
        uchebb=pyfftw.empty_aligned((2*self.N[0]-2,self.N[1], self.N[2],self.N[3],1), dtype='float64')
        uchebf=pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],self.N[3],1), dtype='complex128')
        self.cheb_obj=pyfftw.FFTW(uchebb,uchebf,axes=(0,),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=threads)
        #pressure
        Pchebb=pyfftw.empty_aligned((2*self.N[0]-2,self.N[1], self.N[2],1,1), dtype='float64')
        Pchebf=pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],1,1), dtype='complex128')
        self.Pcheb_obj=pyfftw.FFTW(Pchebb,Pchebf,axes=(0,),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=threads)
        # Store iCheb things
        #velocity
        uichebb=pyfftw.empty_aligned((2*self.N[0]-2,self.N[1], self.N[2],self.N[3],1), dtype='float64')
        uichebf=pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],self.N[3],1), dtype='complex128')
        self.icheb_obj=pyfftw.FFTW(uichebf, uichebb,axes=(0,),flags=('FFTW_MEASURE',),direction='FFTW_BACKWARD',threads=threads)
        #pressure
        Pichebb=pyfftw.empty_aligned((2*self.N[0]-2,self.N[1], self.N[2],1,1), dtype='float64')
        Pichebf=pyfftw.empty_aligned((self.N[0],self.N[1], self.N[2],1,1), dtype='complex128')
        self.Picheb_obj=pyfftw.FFTW(Pichebf, Pichebb,axes=(0,),flags=('FFTW_MEASURE',),direction='FFTW_BACKWARD',threads=threads)

    def convert(self,u,ufft):
        Nxp=int(self.N[1]/2)
        Nzp=int(self.N[2]/2)
        ufft[:,:,:Nzp+1,:]=u
        ufft[:,0,self.N[2]-Nzp+1:,:]=np.conjugate(np.flip(u[:,0,1:Nzp,:],axis=1))
        ufft[:,Nxp,self.N[2]-Nzp+1:,:]=np.conjugate(np.flip(u[:,Nxp,1:Nzp,:],axis=1))
        ufft[:,self.N[1]-Nxp+1:,self.N[2]-Nzp+1:,:]=np.conjugate(np.flip(np.flip(u[:,1:Nxp,1:Nzp,:],axis=1),axis=2))
        ufft[:,1:Nxp,self.N[2]-Nzp+1:,:]=np.conjugate(np.flip(np.flip(u[:,self.N[1]-Nxp+1:,1:Nzp,:],axis=1),axis=2))
        return ufft

    # These are the functions used for moving between real, fourier, and chebyshev space.
    def fft(self,u,threads=-1):
        if u.shape[3]==3: #velocity
            ufft=self.ufft_object(u)
            self.ufft_temp=self.convert(ufft,self.ufft_temp)
            ufft=1/self.N[1]*1/self.N[2]*self.ufft_temp
        else: #pressure
            ufft=self.Pfft_object(u)
            self.Pfft_temp=self.convert(ufft,self.Pfft_temp)
            ufft=1/self.N[1]*1/self.N[2]*self.Pfft_temp
        return ufft
    def ifft(self,u,threads=-1):
        if u.shape[3]==3: #velocity
            uifft=self.uifft_object(u[:,:,:int(self.N[2]/2)+1,:])
            uifft=self.N[1]*self.N[2]*uifft
        else: #pressure
            uifft=self.Pifft_object(u[:,:,:int(self.N[2]/2)+1,:])
            uifft=self.N[1]*self.N[2]*uifft
        return uifft
    def cheb(self,u,threads=-1):
        # modify field for cheb
        utemp=np.copy(u)+1j*0
        utemp[0,:]=utemp[0,:]/2
        utemp[-1,:]=utemp[-1,:]/2

        # dct
        utemp2=np.concatenate((utemp,np.flip(utemp[1:-1,:,:,:,:],axis=0)),axis=0)
        if u.shape[3]==3: #velocity
            ucheb=np.copy(self.cheb_obj(np.real(utemp2)))+1j*self.cheb_obj(np.imag(utemp2))
        else: #pressure
            ucheb=np.copy(self.Pcheb_obj(np.real(utemp2)))+1j*self.Pcheb_obj(np.imag(utemp2))

        # modify the field
        ucheb=(ucheb+utemp[0,:])*(-1)**np.arange(0,self.N[0])[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]+utemp[-1,:]
        ucheb=ucheb/(self.N[0]-1)
        ucheb[0,:]=ucheb[0,:]/2
        ucheb[-1,:]=ucheb[-1,:]/2
        return ucheb
    def icheb(self,u,threads=-1):
        # modify the field for icheb
        utemp=np.copy(u)+1j*0
        utemp[-1,:]=utemp[-1,:]*2
        utemp[0,:]=utemp[0,:]*2
        utemp=utemp*(self.N[0]-1)
        u0=np.sum(u*(-1)**np.arange(0,self.N[0])[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis],axis=0)
        uN=np.sum(u,axis=0)
        utemp=(utemp-uN)*(-1)**np.arange(0,self.N[0])[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]-u0

        # dict
        if u.shape[3]==3: #velocity
            uicheb=np.copy(self.icheb_obj(np.real(utemp)))+1j*self.icheb_obj(np.imag(utemp))
        else: #pressure
            uicheb=np.copy(self.Picheb_obj(np.real(utemp)))+1j*self.Picheb_obj(np.imag(utemp))
        uicheb=uicheb[:35,:,:,:,:]

        #modify the field
        uicheb[-1,:]=uN
        uicheb[0,:]=u0
        return uicheb
    # Basic derivative functions
    #inp[0,1] -> [fft,real]
    def dx(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to fft if it isn't
        if inpxz==1:
            a=self.fft(a)
        # Calculate the derivative (K is for if it isn't u,v, and w)
        [_,_,_,K,_]=a.shape
        dadx=1j*self.k[0][:,:,:,0:K,np.newaxis]*(2*math.pi/(self.L[1]))*a
        # Return derivative
        if outxz==1:
            dadx=self.ifft(dadx)

        # Return cheb or real
        if inpy==0 and outy==1:
            dadx=self.icheb(dadx)
        if inpy==1 and outy==0:
            dadx=self.cheb(dadx)

        return dadx

    def dz(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to fft if it isn't
        if inpxz==1:
            a=self.fft(a)
        # Calculate the derivative (K is for if it isn't u,v, and w)
        [_,_,_,K,_]=a.shape
        dadz=1j*self.k[1][:,:,:,0:K,np.newaxis]*(2*math.pi/(self.L[2]))*a
        # Return derivative
        if outxz==1:
            dadz=self.ifft(dadz)

        # Return cheb or real
        if inpy==0 and outy==1:
            dadz=self.icheb(dadz)
        if inpy==1 and outy==0:
            dadz=self.cheb(dadz)
            
        return dadz
    
    #inp[0,1] -> [cheb,real]
    def dy(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to cheb if it isn't
        if inpy==1:
            a=self.cheb(a)
        # Get the derivative
        dady=np.zeros(a.shape,dtype=np.complex128)
        dady[:-1,:,:,:,:]=np.polynomial.chebyshev.chebder(a)
        # Return derivative
        if outy==1:
            dady=self.icheb(dady)

        # Return fft or real
        if inpxz==0 and outxz==1:
            dady=self.ifft(dady)
        if inpxz==1 and outxz==0:
            dady=self.fft(dady)

        return dady

    #inp[0,1] -> [fft,real]
    def dx2(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to fft if it isn't
        if inpxz==1:
            a=self.fft(a)
        # Calculate the derivative (K is for if it isn't u,v, and w)
        [_,_,_,K,_]=a.shape
        dadx=-(self.k[0][:,:,:,0:K,np.newaxis]*(2*math.pi/(self.L[1])))**2*a
        # Return derivative
        if outxz==1:
            dadx=self.ifft(dadx)
        # Return cheb or real
        if inpy==0 and outy==1:
            dadx=self.icheb(dadx)
        if inpy==1 and outy==0:
            dadx=self.cheb(dadx)
        return dadx
    def dz2(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to fft if it isn't
        if inpxz==1:
            a=self.fft(a)
        # Calculate the derivative (K is for if it isn't u,v, and w)
        [_,_,_,K,_]=a.shape
        dadz=-(self.k[1][:,:,:,0:K,np.newaxis]*(2*math.pi/(self.L[2])))**2*a
        # Return derivative
        if outxz==1:
            dadz=self.ifft(dadz)
        # Return cheb or real
        if inpy==0 and outy==1:
            dadz=self.icheb(dadz)
        if inpy==1 and outy==0:
            dadz=self.cheb(dadz)
        return dadz
    #inp[0,1] -> [cheb,real]
    def dy2(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to cheb if it isn't
        if inpy==1:
            a=self.cheb(a)
        # Get the derivative
        dady=np.zeros(a.shape,dtype=np.complex128)
        dady[:-2,:,:,:,:]=np.polynomial.chebyshev.chebder(np.polynomial.chebyshev.chebder(a))
        # Return derivative
        if outy==1:
            dady=self.icheb(dady)
        # Return fft or real
        if inpxz==0 and outxz==1:
            dady=self.ifft(dady)
        if inpxz==1 and outxz==0:
            dady=self.fft(dady)
        return dady

    # Linear and nonlinear functions
    def gradu(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Derivatives
        dx=self.dx(a,inpy,outy,inpxz,outxz); dy=self.dy(a,inpy,outy,inpxz,outxz); dz=self.dz(a,inpy,outy,inpxz,outxz)    
        # Concatenate for gradu
        gradu=np.concatenate((dx[:,:,:,np.newaxis,:,:],dy[:,:,:,np.newaxis,:,:],dz[:,:,:,np.newaxis,:,:]),axis=-3)
        return gradu

    #inpy[0,1] -> [cheb,real] inpxz[0,1] -> [fft,real]
    def uu(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Convert to real if it isn't
        if inpy==0:
            a=self.icheb(a)
        if inpxz==0:
            a=self.ifft(a)
        # Compute uu
        uu=a*a[:,:,:,0:1,:]
        uv=a*a[:,:,:,1:2,:]
        uw=a*a[:,:,:,2:3,:]
        if outy==0:
            uu=self.cheb(uu)
            uv=self.cheb(uv)
            uw=self.cheb(uw)
        if outxz==0:
            uu=self.fft(uu)
            uv=self.fft(uv)
            uw=self.fft(uw)
        uu=uu[:,:,:,:,np.newaxis,:]
        uv=uv[:,:,:,:,np.newaxis,:]
        uw=uw[:,:,:,:,np.newaxis,:]
        uu_out=np.concatenate((uu,uv,uw),axis=-2)    
        return uu_out
    
    def H(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        # Using the skew symmetric form of the nonlinearity
        # Calculate gradu and make real
        gradu=self.gradu(a,inpy,1,inpxz,1)
        # Convert to real
        if inpy==0:
            a=self.icheb(a)
        if inpxz==0:
            a=self.ifft(a)
        # Skew-symmetric
        ugradu=np.einsum('ijkln,ijklmn->ijkmn',a, gradu)
        if outy==0:
            ugradu=self.cheb(ugradu)
        if outxz==0:
            ugradu=self.fft(ugradu)

        return -ugradu

    def Hskew(self,a,inpy=0,outy=0,inpxz=0,outxz=0): # This increases the compute time and marginally improves results (this takes about double the time of the advection computation)
        # Using the skew symmetric form of the nonlinearity
        # Calculate gradu and make real
        gradu=self.gradu(a,inpy,1,inpxz,1)
        # Convert to real
        if inpy==0:
            a=self.icheb(a)
        if inpxz==0:
            a=self.ifft(a)
        # Skew-symmetric
        ugradu=np.einsum('ijkln,ijklmn->ijkmn',a, gradu)
        if outy==0:
            ugradu=self.cheb(ugradu)
        if outxz==0:
            ugradu=self.fft(ugradu)
            
        uu=self.uu(a,1,1,1,1)
        uudx=self.dx(uu[:,:,:,:,0,:],1,outy,1,outxz)+self.dy(uu[:,:,:,:,1,:],1,outy,1,outxz)+self.dz(uu[:,:,:,:,2,:],1,outy,1,outxz)
        H=-.5*(ugradu+uudx)
        return H

    def div(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        dx=self.dx(a[:,:,:,0:1,:],inpy,outy,inpxz,outxz)
        dy=self.dy(a[:,:,:,1:2,:],inpy,outy,inpxz,outxz)
        dz=self.dz(a[:,:,:,2:3,:],inpy,outy,inpxz,outxz)
        div=dx+dy+dz
        return div
    
    def lap(self,a,inpy=0,outy=0,inpxz=0,outxz=0):
        dx2=self.dx2(a,inpy,outy,inpxz,outxz)
        dy2=self.dy2(a,inpy,outy,inpxz,outxz)
        dz2=self.dz2(a,inpy,outy,inpxz,outxz)
        lap=dx2+dy2+dz2
        return lap
