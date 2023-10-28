import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.fitting import mae, seq_fit_parm
from scipy import linalg
from scipy.optimize import curve_fit, basinhopping
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist,plot_first, plot_second
from matplotlib.ticker import FormatStrFormatter
from time import sleep
import sys
import tqdm
import time
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")
from impedance.validation import linKK


def nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess,bounds,f2_mask,ub):
    
        mask=np.array(Z1.imag)<0
        Z1 = Z1[mask]
        f1 = frequency[mask]
        Z2 = Z2[mask]

        mask=np.array(f1)<f2_mask
        Z2=Z2[mask]
        f2=f1[mask]
        Z1_max = max(abs(Z1))
        Z2_max = max(abs(Z2))

        Z1stack = np.hstack([Z1.real/Z1_max, Z1.imag/Z1_max])
        Z2stack = np.hstack([Z2.real/Z2_max, Z2.imag/Z2_max])
        
        Z = np.hstack([Z1stack,Z2stack])
        
        sigma1 = np.ones(len(Z1stack))
        
        sigma2 = np.ones(len(Z2stack))

        sigma = np.hstack([sigma1,sigma2])


        freq=f1

        popt, pcov = curve_fit(lambda f, *p: simul_fit(f,p,circ_str_1,circ_str_2,Z1_max,Z2_max,f2_mask,ub), freq, Z, p0=initial_guess, maxfev = int(1e10),bounds=bounds,sigma=sigma)
        perror = np.sqrt(np.diag(ub*pcov*ub.T))
        p = popt
        p=p*ub
        
        L0 = p[0]
        R0 = p[1]
        prc1 = p[2:7]
        prc2 = p[7:9]
        pt1 = p[9:14]
        pt2 = p[14:]

        f1=f1
        mask = np.array(f1) < f2_mask
        f2 = f1[mask]

        initial_guess_1 = np.hstack((L0,R0,prc1,pt1))

        initial_guess_2 = np.hstack((pt1,pt2,prc1,prc2))

        circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)

        Z1_fit = circuit_1.predict(f1)

        circuit_2 = CustomCircuit(circ_str_2, initial_guess=initial_guess_2)
        Z2_fit = circuit_2.predict(f2)
        maeeis = mae(Z1,Z1_fit)
        maenleis = mae(Z2,Z2_fit)


        plot_first(ax1, Z1, fmt='o')
        plot_first(ax1, Z1_fit, fmt='-',lw=3)
        ax1.legend(['Data','Model'],fontsize=16,loc='upper left')
        plot_second(ax2, Z2, fmt='o')
        plot_second(ax2, Z2_fit, fmt='-',lw=3)
        ax2.legend(['Data','Model'],fontsize=16,loc='upper left')
        return(Z1,Z1_fit,Z2,Z2_fit,maeeis,maenleis,p,perror)
def simul_fit(f,p,circ_str_1,circ_str_2,Z1_max,Z2_max,f2_mask,ub):
    p = p*ub

    L0 = p[0]
    R0 = p[1]
    prc1 = p[2:7]
    prc2 = p[7:9]
    pt1 = p[9:14]
    pt2 = p[14:]

    f1=f
    mask = np.array(f1) < f2_mask
    f2 = f1[mask]

    initial_guess_1 = np.hstack((L0,R0,prc1,pt1))

    initial_guess_2 = np.hstack((pt1,pt2,prc1,prc2))

    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_2 = CustomCircuit(circ_str_2, initial_guess=initial_guess_2)
    Z1 = circuit_1.predict(f1)
    Z2 = circuit_2.predict(f2)
    Z1 = Z1/Z1_max
    Z2 = Z2/Z2_max
    Z1stack = np.hstack([Z1.real, Z1.imag])
    Z2stack = np.hstack([Z2.real, Z2.imag])
    return (np.hstack([Z1stack,Z2stack]))


def linear_fit (ax,Z1,frequency,circ_str_1,bounds_1,initial_guess_1):
    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    sigma = np.ones(len(Z1_f)*2)*max(abs(Z1_f))
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1,sigma = sigma)
    # circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)


    Z1f_fit = circuit_1.predict(f1_f)
    plot_first(ax, Z1_f, fmt='o')
    plot_first(ax, Z1f_fit, fmt='-',lw=3)
    ax.legend(['Data','Model'],fontsize=16,loc='upper left')

    err = mae(Z1_f,Z1f_fit)
    p = circuit_1.extract()
    p0 = list(p.values())
    return(Z1_f,Z1f_fit,err,p0,p,circuit_1.conf_)

def nonlinear_fit (ax,Z2,frequency,circ_str_1,bounds_1,initial_guess_1,const):
    mask = np.array(frequency)<10
    Z2_f=Z2[mask]
    f2_f=frequency[mask]
    sigma = np.ones(len(Z2_f)*2)*max(abs(Z2_f))
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1,constants = const)
    circuit_1.fit(f2_f, Z2_f,bounds = bounds_1,sigma = sigma)
    Z2f_fit = circuit_1.predict(f2_f)
    plot_second(ax, Z2_f, fmt='o')
    plot_second(ax, Z2f_fit, fmt='-',lw=3)
    ax.legend(['Data','Model'],fontsize=16,loc='upper left')

    err = mae(Z2_f,Z2f_fit)
    p = circuit_1.extract()
    p0 = list(p.values())
    return(Z2_f,Z2f_fit,err,p0,circuit_1.conf_)


def cost_seq(data,model):
    Max = max(abs(data))
    data = data/Max
    model = model/Max
    sum1 = np.sum((data.real-model.real)**2)
    sum2 = np.sum((data.imag-model.imag)**2)
    return(sum1+sum2)


def fun_1 (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,1e-2,1e-2,1e-2,.1,1e-4,10,5e-2,1e-2,10,1e-2,10000]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[-1,0],pf[7:],[-1,0]])/ub    
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))

def fun_10percent_f (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,1e-3,1e-3,1e-3,1e-3,1e-5,.1,5e-3,1e-3,10,1e-2,100]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[-1e1,0],pf[7:],[10,0.1]])/ub    
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))

def fun_10percent_a (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,.15,1e-3,.1e-3,1,1e-2,10
                       ,1e-2,2e-2,10,1e-2,10]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[0,0],pf[7:],[0,0]])/ub    
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))

def fun_30percent_f (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,1e-3,1e-3,1e-3,1e-3,1e-5,.1,5e-3,1e-3,10,1e-2,100]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[-1e2,0],pf[7:],[1e1,0]])/ub   
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))

def fun_30percent_a (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,1e-3,1e-3,1e-3,1e-3
                       ,1e-2,1000
                       ,5e-3,1e-3,10,1e-2,1000]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[0,0],pf[7:],[0,0]])/ub   

    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))


def fun_50percent (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):

    initial_guess_1 = [1e-7,1e-3,1e-3,1e-3,1e-3
                       ,1e-5,.1
                       
                       ,5e-3,1e-3,10
                       
                      ,1e-2,100]


    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[0,0],pf[7:],[0,0]])/ub    
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))

def fun_cc (ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,bounds_1,bounds_2,ub):
    initial_guess_1 = [1e-7,1e-3,1e-3,1e-3,1e-3,1e-5,.1,5e-3,1e-3,10,1e-2,100]

    mask=np.array(Z1.imag)<0
    Z1_f=Z1[mask]
    f1_f=frequency[mask]
    circuit_1 = CustomCircuit(circ_str_1, initial_guess=initial_guess_1)
    circuit_1.fit(f1_f, Z1_f,bounds = bounds_1)
    Z1f_fit = circuit_1.predict(f1_f)
    p=circuit_1.extract()
    pf = list(p.values())
    initial_guess_a = np.hstack([pf[0:7],[-1e2,0],pf[7:],[1e1,0]])/ub   
    return(nleis(ax1,ax2,Z1,Z2,frequency,circ_str_1,circ_str_2,initial_guess_a,bounds_2,10,ub))
