import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import check_and_eval,rmse
from .fitting import set_default_bounds,buildCircuit,calculateCircuitLength,extract_circuit_elements
from scipy.optimize import minimize
ints = '0123456789'

def data_processing(f,Z1,Z2,max_f=10):
    mask = np.array(Z1.imag)<0
    f = f[mask]
    Z1 = Z1[mask]
    Z2 = Z2[mask]
    Z2_same_dim = Z2
    mask1 = np.array(f)<max_f
    f2 = f[mask1]
    Z2 = Z2 [mask1]
    return (f,Z1,f2,Z2,Z2_same_dim)

def simul_fit(frequencies, Z1, Z2, circuit_1,circuit_2, initial_guess, constants={},
                bounds = None, opt='max',param_norm = True, max_f=10,positive = True,
                **kwargs):

    """ Main function for the simultaneous fitting of EIS and NLEIS edata.

    By default, this function uses `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    to fit the equivalent circuit. 

    Parameters
    -----------------
    frequencies : numpy array
        Frequencies

    Z1 : numpy array of dtype 'complex128'
        EIS
    Z1 : numpy array of dtype 'complex128'
        NLEIS

    circuit_1 : string
        String defining the EIS equivalent circuit to be fit
    circuit_2 : string
        String defining the NLEIS equivalent circuit to be fit

    initial_guess : list of floats
        Initial guesses for the fit parameters

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to bounds on all
        parameters of 0 and np.inf, except the CPE alpha
        which has an upper bound of 1

    opt : str, optional
        Default is max normalization. Other normalization will be supported
        in the future 

    global_opt : bool, optional
        If global optimization should be used (uses the basinhopping
        algorithm). Defaults to False
    max_f: int
        The the maximum frequency of interest for NLEIS
    global_opt : bool, optional
        Defaults to True for only positive nyquist plot

    kwargs :
        Keyword arguments passed to scipy.optimize.curve_fit or
        scipy.optimize.basinhopping

    Returns
    ------------
    p_values : list of floats
        best fit parameters for EIS and NLEIS data

    p_errors : list of floats
        one standard deviation error estimates for fit parameters

    """

    if constants is not None:
        constants_1 = {}
        constants_2 = {}
        for name in constants:
            if name[3] != 'n':
                constants_1[name]=constants[name]
                constants_2[name[0:3]+'n'+name[3:]]=constants[name]
            if name[3] == 'n':
                constants_1[name[0:3]+name[4:]]=constants[name]
                constants_2[name]=constants[name]
    else:
        constants_1 = {}
        constants_2 = {}
        
    ##    # set upper and lower bounds on a per-element basis

    if bounds is None:
        edit_circuit = ''
        i=0
        while i < len(circuit_1):
            if circuit_1[i] == 'T' :
                edit_circuit += circuit_1[i:i+3]+'n'
                i+=3
            elif circuit_1[i:i+2]=='RC':
                edit_circuit += circuit_1[i:i+3]+'n'
                i+=3
            else:
                edit_circuit += circuit_1[i]
                i+=1
        bounds = set_default_bounds(edit_circuit, constants=constants_2)
        ub = np.ones(len(bounds[1]))
    else:
        if param_norm:
            
            ub = bounds[1]
            bounds = bounds/ub
        else:
            ub = np.ones(len(bounds[1]))
            
    initial_guess = initial_guess/ub


    if positive:
        mask1 = np.array(Z1.imag)<0
        frequencies = frequencies[mask1]
        Z1 = Z1[mask1]
        Z2 = Z2[mask1]
        mask2 = np.array(frequencies)<max_f
        Z2 = Z2[mask2] 
    else:
        mask2 = np.array(frequencies)<max_f
        Z2 = Z2[mask2] 

    Z1stack = np.hstack([Z1.real, Z1.imag])
    Z2stack = np.hstack([Z2.real, Z2.imag])
    Zstack = np.hstack([Z1stack,Z2stack])
    # weighting scheme for fitting
    if opt == 'max':
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e5
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13
        Z1max = max(np.abs(Z1))
        Z2max = max(np.abs(Z2))
        
        sigma1 = np.ones(len(Z1stack))*Z1max
        sigma2 = np.ones(len(Z2stack))*Z2max
        kwargs['sigma'] = np.hstack([sigma1, sigma2])

        popt, pcov = curve_fit(wrapCircuit_simul(circuit_1, constants_1,circuit_2,constants_2,ub,max_f), frequencies,
                           Zstack,
                           p0=initial_guess, bounds=bounds, **kwargs)
    

    # Calculate one standard deviation error estimates for fit parameters,
    # defined as the square root of the diagonal of the covariance matrix.
    # https://stackoverflow.com/a/52275674/5144795
        perror = np.sqrt(np.diag(ub*pcov*ub.T))

        return popt*ub, perror
    if opt == 'neg':
        bounds = tuple(tuple((bounds[0][i], bounds[1][i])) for i in range(len(bounds[0])))

        res = minimize(warpNeg_log_likelihood(frequencies,Z1,Z2,circuit_1, constants_1,circuit_2,constants_2,ub,max_f), x0=initial_guess,bounds=bounds,**kwargs)
        
        return (res.x*ub,None)
        
    #     # weighting scheme for fitting
    #     if Normalization == 'max':
    #         Z1max = max(np.abs(Z1))
    #         Z2max = max(np.abs(Z2))
            
    #         sigma1 = np.ones(len(Z1stack))*Z1max
    #         sigma2 = np.ones(len(Z2stack))*Z2max
    #         kwargs['sigma'] = np.hstack([sigma1, sigma2])

    #     popt, pcov = curve_fit(wrapCircuit_simul(circuit_1, constants_1,circuit_2,constants_2,ub,max_f), frequencies,
    #                            Zstack,
    #                            p0=initial_guess, bounds=bounds, **kwargs)
        

    #     # Calculate one standard deviation error estimates for fit parameters,
    #     # defined as the square root of the diagonal of the covariance matrix.
    #     # https://stackoverflow.com/a/52275674/5144795
    #     perror = np.sqrt(np.diag(ub*pcov*ub.T))

    # return popt*ub, perror
    

def warpNeg_log_likelihood(frequencies,Z1,Z2,circuit_1, constants_1,circuit_2,constants_2,ub,max_f=10):
    
    def warppedNeg_log_likelihood(parameters):
        f1 =frequencies
        mask = np.array(frequencies)<max_f
        f2 = frequencies[mask]
        x1,x2 = wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters*ub)
        log1 = np.log(sum((Z1-x1)**2))
        log2 = np.log(sum((Z2-x2)**2))
        return(log1+log2)
    return warppedNeg_log_likelihood
        
        


def wrapCircuit_simul(circuit_1, constants_1,circuit_2,constants_2,ub,max_f=10):
    """ wraps function so we can pass the circuit string """
    def wrappedCircuit_simul(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit_1 : string        
        constants_1 : dict
        circuit_2 : string        
        constants_2 : dict
        max_f: int
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """
        
        f1 =frequencies
        mask = np.array(frequencies)<max_f
        f2 = frequencies[mask]
        x1,x2 = wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters*ub)

        y1_real = np.real(x1)
        y1_imag = np.imag(x1)
        y1_stack = np.hstack([y1_real, y1_imag])
        y2_real = np.real(x2)
        y2_imag = np.imag(x2)
        y2_stack = np.hstack([y2_real, y2_imag])

        return np.hstack([y1_stack, y2_stack])
    return wrappedCircuit_simul

def wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters):
    '''

    Parameters
    ----------
    circuit_1 : string
    constants_1 : dict
    circuit_2 : string
    constants_2 : dict
    f1 : list of floats
    f2 : list of floats
    parameters : list of floats


    Returns
    -------
    Z1 and Z2

    '''

    p1,p2 = individual_parameters(circuit_1,parameters)

    x1 = eval(buildCircuit(circuit_1, f1, *p1,
                          constants=constants_1, eval_string='',
                          index=0)[0],
             circuit_elements)
    x2 = eval(buildCircuit(circuit_2, f2, *p2,
                          constants=constants_2, eval_string='',
                          index=0)[0],
             circuit_elements)
    return(x1,x2)
def individual_parameters(circuit,parameters):
    if circuit == '':
        return [],[]
    parameters = list(parameters)
    elements_1 = extract_circuit_elements(circuit)
    p1 = []
    p2 = []
    indx = 0
    for element in elements_1:
        if element[0]=='T' or element[0:2] =='RC' :
            raw_element = get_element_from_name(element)
            num_params = check_and_eval(raw_element).num_params
            p1+=parameters[indx:indx+num_params]
            p2+=parameters[indx:indx+2+num_params]
            indx+=2+num_params

        else:
            raw_element = get_element_from_name(element)
            num_params = check_and_eval(raw_element).num_params
            p1+=parameters[indx:indx+num_params]
            indx+=num_params
    return p1,p2









