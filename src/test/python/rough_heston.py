#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 08:35:48 2021

@author: caleb
"""
import numpy as np
import scipy.special as sp

i = complex(0, 1)

def fractional_riccati(x, u, Lambda, gamma, rho, theta, v0, alpha):
    section_1 = 0.5 * (-u**2 - i*u)
    section_2 = Lambda * (i*u*rho*gamma- 1) * x
    section_3 = np.power(gamma*Lambda*x, 2)/2
    
    return section_1 + section_2 + section_3

def phi(u, T, Lambda, gamma, rho, theta, v0, alpha):
    N = 1000
    dt = T/N

    # Initial condition(h_a_0 is 0) 
    h_a_tk = np.zeros(N+1, dtype = np.complex_)
    
    # Weights of the Corrector Predictor formulas
    a = []
    b = []
    
    # For each tk calculate the weights aj,k+1 and bj,k+1 
    for k in range(N):
        
        # Calculate the a0,k+1
        aj0 = [(dt**alpha) * ((k)**(alpha + 1) - ((k-alpha)*(k+1)**alpha))/sp.gamma(alpha + 2)]
        
        # Calculate aj,k+1
        aj = [dt**alpha * ((k - j + 2)**(alpha + 1) + 
                           ((k - j)**(alpha + 1)) - 
                           2 * (k - j + 1)**(alpha + 1))/sp.gamma(alpha + 2) for j in range(k)]
        
        # Extend aj0 by aj
        aj0.extend(aj)
        
        # Add numpy array to list of weights a
        a.append(np.array(aj0))
        
        bj = [dt**alpha * ((k - j + 1)**(alpha) - (k - j)**alpha) for j in range(k+1)]
        
        b.append(np.array(bj))
        
        
    # Declare values of F(a, h(a, tk))
    F_a_h_a_tk = np.zeros(N+1, dtype = np.complex_)
    F_a_h_a_tk[0] = fractional_riccati(h_a_tk[0], u, Lambda, gamma, rho, theta, v0, alpha)
    
    # Use Adam's scheme to calculate the value of the integrals g1 and g2 in the characteristic function
    for j in range(1, N+1):
        # Calculate the sum used to find p
        hp_sum = b[j-1].dot(F_a_h_a_tk[:j])
        hp = (dt ** alpha) * hp_sum/sp.gamma(alpha + 1)
        
        # Calculate first part of h_a_tk
        h_a_tk_first_part = a[j-1].dot(F_a_h_a_tk[:j])
        h_a_tk_second_part = np.sum(F_a_h_a_tk[:j]) * (dt**alpha)/sp.gamma(alpha + 2)
        
        h_a_tk[j] =  h_a_tk_first_part + h_a_tk_second_part
        
        F_a_h_a_tk[j] = fractional_riccati(h_a_tk[j-1], u, Lambda, gamma, rho, theta, v0, alpha) 
        
        h_a_tk_first_part = (fractional_riccati(p, u, Lambda, gamma, rho, theta, v0, alpha) + 
                            ((j-1)**(alpha+1) - (j-1-alpha)*(j**alpha)) * F_a_h_a_tk[0])
        
        # Calculate the sum at the end of h_a_tk
        h_a_tk_sum = np.flip(a[1:j-1]).dot(F_a_h_a_tk[1:j-1])
        
        # Join the two sums
        h_a_tk[j] = (h_a_tk_first_part+ h_a_tk_sum) * (dt**alpha)/sp.gamma(alpha + 2) 
    
        # Update value of F_a_h_a_tk
        F_a_h_a_tk[j] = fractional_riccati(h_a_tk[j], u, Lambda, gamma, rho, theta, v0, alpha) 
        
    # Calculate g1 using trapezoidal weights
    w1 = np.append([0, 0.5], np.full(N-2, 1.0))
    w1 = np.append(w1, [0.5])
    
    # Calculate the value of g1
    g1 = dt * w1.dot(h_a_tk)
    
    # Calculate Adam's weights for g2 fractional integral
    w_adams = np.full(N+1, 1.0)
    
    # Calculate the value of w0
    w_adams[0] = (N-1)**(2 - alpha) - (N - 2 + alpha)* N**(1-alpha)
    
    # Will be used to calculate Adams weights
    vector_N = np.arange(N, 1, -1)
    
    # We use 2 - alpha = 1 + (1 - alpha)
    w_adams[1:N] = vector_N**(2 - alpha) + (vector_N-2)**(2-alpha) - 2 * (vector_N-1)**(2 - alpha)
    
    # Divide weights by 1/(Γ(2 + (1 -alpha)))
    w_adams = (dt**(1 -alpha))/sp.gamma(3-alpha) * w_adams
    
    # Multiply weights with y to get g2
    g2 = w_adams.dot(h_a_tk)
    
    # Return characteristic function
    return np.exp(theta * gamma * g1 + v0 * g2)
           
def rough_heston(S0, K, T, r, Lambda, gamma, rho, theta, v0, alpha):
    integral, iterations, max_number = 0, 1000, 100
    du = max_number/iterations
    
    # Midpoint rule for complex integral
    for j in range(1, iterations):
        u = du * (2*j-1)/2
        u_i = u - i/2
        
        element_1 = phi(u_i, T, Lambda, gamma, rho, theta, v0, alpha)
        element_2 = np.exp(i * np.log(K) * u).real/(0.25 + u**2)

        integral += (element_1 * element_2).real
        
    # Price using the Lewis formula
    price = S0 - ((np.sqrt(S0 * K) * np.exp(-r * T/2)/np.pi) * integral)
    return price

S0, K, T, r, Lambda, gamma, rho, theta, v0, alpha = 120, 100, 1, 0.01, 2, 0.05, -0.5, 0.04, 0.4, 0.6
price = rough_heston(S0, K, T, r, Lambda, gamma, rho, theta, v0, alpha)  
print(price)
    
 
    
 
    
 
    
 
    
 
    
 
    
    
