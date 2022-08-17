# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 22:56:24 2022

@author: Andreis Maxime 

"""

import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
from EEG_Multi_Electrode_Treatment_def import *
import scipy.io
import time



#Parameters for KMD 
alpha = 100
omega_wavelet = 1

#Name of the data to treat
file_name = 'ID02_233h'

data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['EEG']

# #For short term data
# number_of_signal = len(datatot[0])
# datatot = np.transpose(np.asarray(datatot))
                       
#For long term data
datatot = np.asarray(datatot)
number_of_signal = len(datatot)

for k in range(number_of_signal):
    data.append(np.transpose(datatot[:][k]).tolist())
    
signal_size = len(data[0])
    
# #Decomment for 1024 Hz signals 
# #Downsampling the signal based on the downscale factor
# downscale = 2 
# for k in range(number_of_signal):
#     data[k] = downsampling(data[k],downscale)
# signal_size = len(data[0])

#Size of the Kernel
kernel_lenght = 100


#Creating the fonction database
func_datab=[]
#Fixed parameters
a = 0.2
c = 0 
theta = 0
#Ranging parameters 
b_var = np.linspace(-0.3,0,10)
omega_var = np.linspace(3,7,10) * np.pi
tau_var = np.linspace(0.05,0.95,6)


for tau in tau_var:
    for omega in omega_var:
        for b in b_var:
            func_datab.append(Spikes_function(omega,tau,theta,a,b,c,alpha))
            

#creation of the Kernel
time_mesh = np.transpose(np.linspace(0,1,kernel_lenght).reshape((kernel_lenght, 1)))
Kmode = np.zeros((np.size(time_mesh),np.size(time_mesh)))
for i in range(len(func_datab)):
    print(i)
    Kmode = Kmode + createKernel(time_mesh,func_datab[i])
sigma = np.max(Kmode) * 0.05
Knoise = createNoisekernel(time_mesh,sigma)
Ktot = Kmode + Knoise


#Parameters for the alignement energy calculation
jump = 200 #Number of point beetween each alignment energy calculation
electrode_number = 1 #Number of the electrode that should be treated 

#Parameters for mean value and variance calculation :
Z = 50 #Number of values, before and after, used to calculate the mean value and the variance

#Parameter for the detection (* variance mean value)
Threshold = 4 


start_time = time.perf_counter()
#Creation of the excel for the result 
result_excel = pd.DataFrame(columns = ['Signal','Spike Location'] )
    
#Detection on each electrode of the dataset
for i in range(number_of_signal):
    print(i)
    result = variance_detection(i,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size,Threshold)
    for k in range(len(result)):
        result_excel.loc[len(result_excel)] = result[k]

#Excel output with the results of the detection
file_name_excel = 'resultats '+file_name+'.xlsx'
result_excel.to_excel(file_name_excel)
    
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
