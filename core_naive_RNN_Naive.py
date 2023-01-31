# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:00:58 2023

@author: arredond
"""

"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2023
"""


# %%   
"""
Import modules

"""


import tensorflow as tf
from tensorflow.keras import layers
    
    

class RNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.state_size = [[1],[1]]
        self.output_size = [[1]]
     
        self.ls = [layers.Dense(32, 'softplus')]
        self.ls += [layers.Dense(2)]

        
    def call(self, inputs, states):
        
        #   states are the internal variables
        
        #   n: current time step, N: old time step
                
        eps_n = inputs[0]
        hs = inputs[1]
        E = 2
        eta = 1
        E_infty = 0.5
        
        gamma_N = states[0]
        eps_N = states[1]
        
        gamma_dot_N = E/eta * (eps_N- gamma_N)
        gamma_n = gamma_N + hs * gamma_dot_N
             
        sig_n = E_infty * eps_n +E*(eps_n-gamma_N)
                
                          
        return sig_n , [gamma_n, eps_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        # define initial values of the internal variables
                
        return [tf.zeros([batch_size, 1]), tf.zeros([batch_size, 1])]


def main(**kwargs):
    
    eps = tf.keras.Input(shape=[None, 1],name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')
        
    cell = RNNCell()
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))


    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    return model



