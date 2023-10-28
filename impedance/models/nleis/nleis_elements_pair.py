import numpy as np
from scipy.special import iv
from impedance.models.circuits.elements import element,circuit_elements,ElementError,OverwriteError



def t(NLEIS):
    """ adds elements in second harmonic

    Notes
    ---------
    .. math::

        Z = \\frac{1}{\\frac{1}{Z_1} + \\frac{1}{Z_2} + ... + \\frac{1}{Z_n}}

     """
    z = len(NLEIS[0])*[0 + 0*1j]
    z += NLEIS[0]
    z += -NLEIS[-1]
    return z
# manually add parallel and series operators to circuit elements w/o metadata
# populated by the element decorator -
# this maps ex. 'R' to the function R to always give us a list of
# active elements in any context
circuit_elements['t'] = t

@element(num_params=3, units=['Ohms', 'Ohms', 'F'])
def TPO(p, f):
    """ A macrohomogeneous porous electrode model from Paasch et al. [1]


    [1] G. Paasch, K. Micka, and P. Gersdorf,
    Electrochimica Acta, 38, 2653–2662 (1993)
    `doi: 10.1016/0013-4686(93)85083-B
    <https://doi.org/10.1016/0013-4686(93)85083-B>`_.
    """

    omega = 2*np.pi*np.array(f)

    Rpore, Rct, Cdl = p[0], p[1], p[2]

    beta = (1j*omega*Rpore*Cdl+Rpore/Rct)**(1/2)

    Z = Rpore/(beta*np.tanh(beta))
    return Z

@element(num_params=4, units=['Ohms', 'Ohms', 'F',''])
def TPOn(p, f):
    """ NLEIS: A macrohomogeneous porous electrode model 



    TPn_0: Rpore
    TPn_1: Rct
    TPn_2: Cdl
    TPn_3: e
    """

    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, e = p[0], p[1], p[2],p[3]
    b1 = (1j*omega*Rpore*Cdl+Rpore/Rct)**(1/2)
    b2 = (1j*2*omega*Rpore*Cdl+Rpore/Rct)**(1/2)

    f=96485.3321233100184/(8.31446261815324*298)
    sinh1 = []
    for x in b1:
        if x < 100:
            sinh1.append(np.sinh(x))
        else:
            sinh1.append(1e10)
    sinh2 = []
    cosh2 = []
    for x in b1:
        if x < 100:
            sinh2.append(np.sinh(2*x))
            cosh2.append(np.cosh(2*x))
        else:
            sinh2.append(1e10)
            cosh2.append(1e10)
    sinh3 = []
    cosh3 = []
    for x in b2:
        if x < 100:
            sinh3.append(np.sinh(x))
            cosh3.append(np.cosh(x))
        else:
            sinh3.append(1e10)
            cosh3.append(1e10)

    mf = ((Rpore**3)/Rct)*e*f/((b1*np.array(sinh1))**2)
    part1 = (b1/b2)*np.array(sinh2)/((b2**2-4*b1**2)*np.tanh(b2))
    part2 = -np.array(cosh2)/(2*(b2**2-4*b1**2))-1/(2*b2**2)
    Z = mf*(part1+part2)

    return Z

@element(num_params=5, units=['Ohms', 'Ohms', 'F','Ohms','s'])
def TDC(p, f):
    """ EIS: A macrohomogeneous porous electrode model with cylindrical diffusion 
    p0=Rpore
    p1=Rct
    p2=Cdl
    p3=Aw
    p4=omegaD

    """
    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2],p[3],p[4]
    i01 = []
    i11 = []
    for x in np.sqrt(1j*omega*τd):
        if x < 100:
            i01.append(iv(0,x))
            i11.append(iv(1,x))
        else:
            i01.append(1e20)
            i11.append(1e20)
    Zd = Aw*np.array(i01)/(np.sqrt(1j*omega*τd)*np.array(i11))

    beta = (1j*omega*Rpore*Cdl+Rpore/(Zd+Rct))**(1/2)
    Z = Rpore/(beta*np.tanh(beta))
    return Z
@element(num_params=7, units=['Ohms', 'Ohms', 'F','Ohms','s','1/V',''])
def TDCn(p, f):
    """ NLEIS: A macrohomogeneous porous electrode model with cylindrical diffusion 
    p0=Rpore
    p1=Rct
    p2=Cdl
    p3=Aw
    p4=omegaD
    p5=V
    p6=e

    """

    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd, κ,e = p[0], p[1], p[2],p[3],p[4],p[5],p[6]

    i01 = []
    i11 = []
    for x in np.sqrt(1j*omega*τd):
        if x < 100:
            i01.append(iv(0,x))
            i11.append(iv(1,x))
        else:
            i01.append(1e20)
            i11.append(1e20)
    i02 = []
    i12 = []
    for x in np.sqrt(1j*2*omega*τd):
        if x < 100:
            i02.append(iv(0,x))
            i12.append(iv(1,x))
        else:
            i02.append(1e20)
            i12.append(1e20)
    Zd1 = Aw*np.array(i01)/(np.sqrt(1j*omega*τd)*np.array(i11))
    Zd2 = Aw*np.array(i02)/(np.sqrt(1j*2*omega*τd)*np.array(i12))

    y1 = Rct/(Zd1+Rct)
    y2 = (Zd1/(Zd1+Rct))
    
    b1 = (1j*omega*Rpore*Cdl+Rpore/(Zd1+Rct))**(1/2)
    b2 = (1j*2*omega*Rpore*Cdl+Rpore/(Zd2+Rct))**(1/2)
    
    f=96485.3321233100184/(8.31446261815324*298)
    sinh1 = []
    for x in b1:
        if x < 100:
            sinh1.append(np.sinh(x))
        else:
            sinh1.append(1e10)
    sinh2 = []
    cosh2 = []
    for x in b1:
        if x < 100:
            sinh2.append(np.sinh(2*x))
            cosh2.append(np.cosh(2*x))
        else:
            sinh2.append(1e10)
            cosh2.append(1e10)
    const= -((Rct*κ*y2**2)-Rct*e*f*y1**2)/(Zd2+Rct)
    mf = ((Rpore**3)*const/Rct)/((b1*np.array(sinh1))**2)
    part1 = (b1/b2)*np.array(sinh2)/((b2**2-4*b1**2)*np.tanh(b2))
    part2 = -np.array(cosh2)/(2*(b2**2-4*b1**2))-1/(2*b2**2)
    Z = mf*(part1+part2)

    return Z

@element(num_params=5, units=['Ohms', 'Ohms', 'F','Ohms','s'])
def TDS(p, f):
    """ EIS: A macrohomogeneous porous electrode model with spherical diffusion
    p0=Rpore
    p1=Rct
    p2=Cdl
    p3=Aw
    p4=τd

    """
    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2],p[3],p[4] 

##    Zd = Aw*τd*np.tanh(np.sqrt(1j*omega*τd))/(np.sqrt(1j*omega*τd)-np.tanh(np.sqrt(1j*omega*τd)))
    Zd = Aw*np.tanh(np.sqrt(1j*omega*τd))/(np.sqrt(1j*omega*τd)-np.tanh(np.sqrt(1j*omega*τd)))

    beta = (1j*omega*Rpore*Cdl+Rpore/(Zd+Rct))**(1/2)
    Z = Rpore/(beta*np.tanh(beta))
    return Z
@element(num_params=7, units=['Ohms', 'Ohms', 'F','Ohms','s','1/V',''])
def TDSn(p, f):
    """ NLEIS: A macrohomogeneous porous electrode model with spherical diffusion
    p0=Rpore
    p1=Rct
    p2=Cdl
    p3=Aw
    p4=omegaD
    p5=V
    p6=e

    """

    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd, κ,e = p[0], p[1], p[2],p[3],p[4],p[5],p[6]
##    Zd1 = Aw*τd*np.tanh(np.sqrt(1j*omega*τd))/(np.sqrt(1j*omega*τd)-np.tanh(np.sqrt(1j*omega*τd)))
##    Zd2 = Aw*τd*np.tanh(np.sqrt(1j*2*omega*τd))/(np.sqrt(1j*2*omega*τd)-np.tanh(np.sqrt(1j*2*omega*τd)))
    Zd1 = Aw*np.tanh(np.sqrt(1j*omega*τd))/(np.sqrt(1j*omega*τd)-np.tanh(np.sqrt(1j*omega*τd)))
    Zd2 = Aw*np.tanh(np.sqrt(1j*2*omega*τd))/(np.sqrt(1j*2*omega*τd)-np.tanh(np.sqrt(1j*2*omega*τd)))

    y1 = Rct/(Zd1+Rct)
    y2 = (Zd1/(Zd1+Rct))
    
    b1 = (1j*omega*Rpore*Cdl+Rpore/(Zd1+Rct))**(1/2)
    b2 = (1j*2*omega*Rpore*Cdl+Rpore/(Zd2+Rct))**(1/2)
    
    f=96485.3321233100184/(8.31446261815324*298)
    sinh1 = []
    for x in b1:
        if x < 100:
            sinh1.append(np.sinh(x))
        else:
            sinh1.append(1e10)
    sinh2 = []
    cosh2 = []
    for x in b1:
        if x < 100:
            sinh2.append(np.sinh(2*x))
            cosh2.append(np.cosh(2*x))
        else:
            sinh2.append(1e10)
            cosh2.append(1e10)
    const= -((Rct*κ*y2**2)-Rct*e*f*y1**2)/(Zd2+Rct)
    mf = ((Rpore**3)*const/Rct)/((b1*np.array(sinh1))**2)
    part1 = (b1/b2)*np.array(sinh2)/((b2**2-4*b1**2)*np.tanh(b2))
    part2 = -np.array(cosh2)/(2*(b2**2-4*b1**2))-1/(2*b2**2)
    Z = mf*(part1+part2)

    return Z
@element(num_params=5, units=['Ohms', 'Ohms', 'F','Ohms','s'])
def TDP(p, f):
    """ EIS: A macrohomogeneous porous electrode model with planar diffusion 
    p0 = Rpore
    p1 = Rct
    p2 = Cdl
    p3 = Aw
    p4 = τd
    Planar

    """
    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2],p[3],p[4] 

##    Zd = Aw*τd/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
    Zd = Aw/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))

    beta = (1j*omega*Rpore*Cdl+Rpore/(Zd+Rct))**(1/2)
    Z = Rpore/(beta*np.tanh(beta))

    return Z
@element(num_params=7, units=['Ohms', 'Ohms', 'F','Ohms','s','1/V',''])
def TDPn(p, f):
    """ NLEIS: A macrohomogeneous porous electrode model with planar diffusion 
    p0 = Rpore
    p1 = Rct
    p2 = Cdl
    p3 = Aw
    p4 = τd
    p5 = κ
    p6 = ε
    Planar

    """

    omega = 2*np.pi*np.array(f)
    Rpore, Rct, Cdl, Aw, τd, κ,e = p[0], p[1], p[2],p[3],p[4],p[5],p[6]
##    Zd1 = Aw*τd/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
##    Zd2 = Aw*τd/(np.sqrt(1j*2*omega*τd)*np.tanh(np.sqrt(1j*2*omega*τd)))
    Zd1 = Aw/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
    Zd2 = Aw/(np.sqrt(1j*2*omega*τd)*np.tanh(np.sqrt(1j*2*omega*τd)))

    y1 = Rct/(Zd1+Rct)
    y2 = (Zd1/(Zd1+Rct))
    
    b1 = (1j*omega*Rpore*Cdl+Rpore/(Zd1+Rct))**(1/2)
    b2 = (1j*2*omega*Rpore*Cdl+Rpore/(Zd2+Rct))**(1/2)
    
    f=96485.3321233100184/(8.31446261815324*298)
    sinh1 = []
    for x in b1:
        if x < 100:
            sinh1.append(np.sinh(x))
        else:
            sinh1.append(1e10)
    sinh2 = []
    cosh2 = []
    for x in b1:
        if x < 100:
            sinh2.append(np.sinh(2*x))
            cosh2.append(np.cosh(2*x))
        else:
            sinh2.append(1e10)
            cosh2.append(1e10)
    const= -((Rct*κ*y2**2)-Rct*e*f*y1**2)/(Zd2+Rct)
    mf = ((Rpore**3)*const/Rct)/((b1*np.array(sinh1))**2)
    part1 = (b1/b2)*np.array(sinh2)/((b2**2-4*b1**2)*np.tanh(b2))
    part2 = -np.array(cosh2)/(2*(b2**2-4*b1**2))-1/(2*b2**2)
    Z = mf*(part1+part2)

    return Z
@element(num_params=3, units=['Ohm', 'F'])
def RCO(p,f):
    w=np.array(f)*2*np.pi
    Rct=p[0]
    Cdl=p[1]
    
    Z1r=Rct/(1+(w*Rct*Cdl)**2);
    Z1i=(-w*Cdl*Rct**2)/(1+(w*Rct*Cdl)**2);
    Z1 = Z1r+1j*Z1i

    return(Z2)
@element(num_params=3, units=['Ohm', 'F',''])
def RCOn(p,f):
    w=np.array(f)*2*np.pi
    Rct=p[0]
    Cdl=p[1]
    e=p[2]
    f=96485.3321233100184/(8.31446261815324*298) ## unit in 1/V
    
    Z1r=Rct/(1+(w*Rct*Cdl)**2);
    Z1i=(-w*Cdl*Rct**2)/(1+(w*Rct*Cdl)**2);
    
    tau=w*Rct*Cdl
    
    Z2r=-e*f*(Z1r**2-Z1i**2+4*tau*Z1r*Z1i)/(1+4*tau**2)
    Z2i=e*f*((Z1r**2-Z1i**2)*2*tau-2*Z1r*Z1i)/(1+4*tau**2)
    
    Z2=Z2r+1j*Z2i
    return(Z2)

@element(num_params=4, units=['Ohms', 'F','Ohms','s'])
def RCD(p,f):
    omega=np.array(f)*2*np.pi
    Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3]

##    Zd = Aw*τd/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
    Zd = Aw/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
    tau = omega*Rct*Cdl
    Z = Rct/(Rct/(Rct+Zd)+1j*tau)
    return(Z)
@element(num_params=6, units=['Ohms', 'F','Ohms','s','1/V',''])
def RCDn(p,f):
    omega=np.array(f)*2*np.pi
    Rct, Cdl, Aw, τd, κ,e = p[0], p[1], p[2],p[3],p[4],p[5]

##    Zd1 = Aw*τd/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
##    Zd2 = Aw*τd/(np.sqrt(1j*2*omega*τd)*np.tanh(np.sqrt(1j*2*omega*τd)))
    Zd1 = Aw/(np.sqrt(1j*omega*τd)*np.tanh(np.sqrt(1j*omega*τd)))
    Zd2 = Aw/(np.sqrt(1j*2*omega*τd)*np.tanh(np.sqrt(1j*2*omega*τd)))

    f=96485.3321233100184/(8.31446261815324*298)


    tau = omega*Rct*Cdl
    y1 = Rct/(Zd1+Rct)
    y2 = (Zd1/(Zd1+Rct))
    
    Z1 = Rct/(y1+1j*tau)
    const= ((Rct*κ*y2**2)-Rct*e*f*y1**2)/(Zd2+Rct)

    Z2 = (const*Z1**2)/(2*tau*1j+Rct/(Zd2+Rct))
    
    return(Z2)


@element(num_params=5, units=['Ohm', 'Ohm', 'F','', ''])
def Tsn(p,f):
    """ Second harmonic discrete transmission line model built based on the Randles circuit
        Tsn_0: Rpore
        Tsn_1: Rct
        Tsn_2: Cdl
        Tsn_3: N (number of circuit element)
        Tsn_4: e

    
    """
    
    #p0=p[0:4]
    I=Ti(p[0:4],f) # calculate the current fraction (1st harmonic)

    N=int(p[3])
    w=np.array(f)*2*np.pi
    Rpore=p[0]/N
    Rct=p[1]*N
    Cdl=p[2]/N
    
    e=p[4]
    f=96485.3321233100184/(8.31446261815324*298) ## unit in 1/V
    
    Z1r=Rct/(1+(w*Rct*Cdl)**2);
    Z1i=(-w*Cdl*Rct**2)/(1+(w*Rct*Cdl)**2);
    Z1=Z1r+1j*Z1i
    
    tau=w*Rct*Cdl
    
    Z2r=-e*f*(Z1r**2-Z1i**2+4*tau*Z1r*Z1i)/(1+4*tau**2)
    Z2i=e*f*((Z1r**2-Z1i**2)*2*tau-2*Z1r*Z1i)/(1+4*tau**2)
    
    Z2=Z2r+1j*Z2i
    Z12t=(Rct/(1+(2*w*Rct*Cdl)**2))+1j*((-w*2*Cdl*Rct**2)/(1+(2*w*Rct*Cdl)**2))
    #Z12t=Z12t*0
    if N==1:
        return(Z2)
    
    if N==2:
        sum1=Z1**2/(2*Z1+Rpore)**2
        #sum2=(Z12t*Rpore+Rpore**2)/(2*Z12t+Rpore)**2
        sum2=(Z12t*Rpore+Rpore**2)/((2*Z12t+Rpore)*(2*Z1+Rpore))
        Z=(sum1+sum2)*Z2
        return(Z)
    Z=np.zeros((len(w)),dtype = complex)
    for freq in range(0,len(w)):
        Ii=I[freq]
        
        A = np.arange(N-1,0,-1)
        A1 = np.arange(N-1,0,-1)
        
        for i in range (0,N-2):
            for j in range(0,N-1-i):
                A1[j]=A1[j]-1
            A=np.vstack((A,A1))
        A=A*Rpore 
        A=np.append(A, np.zeros((N-1,1)), axis = 1)
        A=np.append(A, np.zeros((1,N)), axis = 0)
        A2=np.zeros((N-1,N))
        for i in range(0,N-1):
            A2[i,0]+=1
            A2[i,N-1-i]-=1
        A2=np.vstack((A2,np.zeros(N)))
        A2=A2*Z12t[freq]
        
        A3=np.vstack((np.zeros((N-1,N)),np.ones(N)))
        
        Ax = A2+A+A3
        
        b=np.zeros((N,1),dtype = complex)

        for i in range (0,N-1):
            b[i]=Ii[-1]**2-Ii[i]**2
        
        I2=np.linalg.solve(Ax,-b*Z2[freq])
        Z[freq]=Z2[freq]*Ii[0]**2+I2[-1]*Z12t[freq]
    return(Z)

@element(num_params=4, units=['Ohm', 'Ohm', 'F', ''])
def Ti(p,f):
    N=int(p[3])
    w=np.array(f)*2*np.pi
    Rct=p[1]*N
    Cdl=p[2]/N
    Rpore=p[0]/N
    Z1=Rct/(1+(w*Rct*Cdl)**2);
    Z2=(-w*Cdl*Rct**2)/(1+(w*Rct*Cdl)**2);  
    Zran=Z1+Z2* 1j;
    Req=Zran;
    for i in range(1,N):
        
        Req_inv=(1/(Req+Rpore))+1/Zran
        Req=1/Req_inv
        
    Req=Req+Rpore
    
    I=np.zeros((len(w),N),dtype=complex)
    for freq in range (0,len(w)):
        b1=np.ones(N)*Req[freq]

        A=np.identity(N)*Zran[freq]
        
        A1=np.ones((N,N))*Rpore
        
        for i in range(0,N):
            
            A1[i,:]=A1[i,:]*(i+1)
            
            for j in range(0,i):
                
                A[i][j]=-(i-j)*Rpore

        A = A+A1
        
        b = b1
        
        I[freq,:] = np.linalg.solve(A, b)   
    return(I)


def get_element_from_name(name):
    excluded_chars = '0123456789_'
    return ''.join(char for char in name if char not in excluded_chars)


def typeChecker(p, f, name, length):
    assert isinstance(p, list), \
        'in {}, input must be of type list'.format(name)
    for i in p:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, p)
    for i in f:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, f)
    assert len(p) == length, \
        'in {}, input list must be length {}'.format(name, length)
    return
