# General Terms
## Epochs
- one epoch = one forward pass and one backward pass of all the training examples

## Batch Size
- The batch size defines the number of samples that will be propagated through the network simultaneously. 
- The number of samples is often a power of 2, to facilitate memory allocation on GPU.
- The algorithm takes the first 128 samples (from 1st to 128th) from the training dataset and trains the network. Next it takes the second 128 samples (from 129st to ...th) and trains the network again.

