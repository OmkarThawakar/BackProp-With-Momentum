# BackProp-With-Momentum
Implementation of Error BackPropagation Training Algorithm With Momentum 

Momentum term is added during weight updation in Error BackPropagation for acceleration of convergence of learning algorithm.
In this method one extra term is added into the newly updated weight along with error signal term of neuron. this extra term
is nothing but the previous error signal term of neuron weighted by a user defined parameter called alpha(usually anges from 0.1 to 0.8).
