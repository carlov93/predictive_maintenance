# LSTM Networks
## Sequence Models
In a feed-forward there is no state maintstate maintained by the network at all. In sequence models there is some sort of dependence through time between the inputs.Its output could be used as part of the next input, so that information can propogate along as the network passes over the sequence. In RNN you have the problem of a shrinking gradient. A gradient shrinks when its backpropagate through time, so earlier networks doesn't learn so much. Thus these kind of networks only have a small memory. 

## LSTM
A Long-short Term Memory network (LSTM) is a type of recurrent neural network designed to overcome problems of basic RNNs so the network can learn long-term dependencies. Specifically, it tackles vanishing and exploding gradients – the phenomenon where, when you backpropagate through time too many time steps, the gradients either vanish (go to zero) or explode (get very large) because it becomes a product of numbers all greater or all less than one. 
In the case of an LSTM, for each element in the sequence, there is a corresponding hidden state ht, which in principle can contain information from arbitrary points earlier in the sequence.

## Gates of LSTM:
Gates = neural network which decise wheter information pass a gate or not. Gates in LSTM are the sigmoid activation functions because we want a gate to give only positive values and should be able to give us a clear cut answer whether, we need to keep a particular feature (1) or we need to discard that feature (0). <br>
LSTM will have 3 gates:
- $f_t$ : forget gate  
- $i_t$ : input gate 
- $o_t$ : output gate 

## Terminology of Vectors
- $c_{t-1}$ = Previous memory (cell state)
- $h_{t-1}$ = Previous output (predition)
- $x_{t}$ = Current input vector
- $h_{t}$ = Current output (prediction)
- $c_{t}$ = Current memory (cell state)

A LSTM has two “hidden states”: c_t  and h_t . Intuitively, c_t  is the “internal” hidden state that retains important information for longer timesteps, whereas h_t is the “external” hidden state that exposes that information to the outside world.

## Activation Functions in LSTM
Traditionally, LSTMs use the tanh activation function for the activation of the cell state and the sigmoid activation function for the node output. Given their careful design, ReLU were thought to __not__ be appropriate for Recurrent Neural Networks (RNNs) such as LSTM because they can have very large outputs so they might be expected to be far more likely to explode than units that have bounded values.


## Architecture of a LSTM

![](../pictures/lstm_key.png)

### Version A
![](../pictures/lstm_mit_c.png)

![](../pictures/lstm_mit_c_formula.png)


### Version B (same like Prof. Niggemann)
![](../pictures/lstm_ohne_c.png)

![](../pictures/lstm_nigemann.png)

![](../pictures/lstm_ohne_c_formula.png)

