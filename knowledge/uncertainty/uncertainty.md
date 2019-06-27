# Uncertainty
## 1. Why should you care about uncertainty?
- One prominent example is that of high risk applications (building a model that helps doctors decide on the preferred treatment for patients)
- Self-driving cars (model is uncertain if there is a pedestrian on the road --> slow the car down or trigger an alert so the driver can take charge)
- Uncertainty can also help us with out of data examples. If the model wasn’t trained using examples similar to the sample at hand it might be better if it’s able to say “sorry, I don’t know”. 
- The last usage of uncertainty is as a tool for practitioners to debug their model. 

## 2. Types of uncertainty
### Model uncertainty (epistemic uncertainty)
- Let’s say you have a single data point and you want to know which linear model best explains your data. There is no good way to choose between different lines  
- Epistemic uncertainty accounts for uncertainty in the model’s parameter. 
- We are not sure which model weights describe the data best, but given more data our uncertainty decreases. 
- This type of uncertainty is important in high risk applications and when dealing with small and sparse data.

### Data uncertainty (aleatoric uncertainty)
- This uncertainty captures the noise inherent in the observation. 
- Sometimes the world itself is stochastic. Obtaining more data will not help us in that case, because the noise is inherent in the data.
- Aleatoric uncertainty is divided into two types:
   - Homoscedastic uncertainty
   - Heteroscedastic uncertainty

__Homoscedastic uncertainty__
- For every observation (x,y) we have the same variance 
- Identical observation noise for every input point x
- Task-dependant or Homoscedastic uncertainty is aleatoric uncertainty which is not dependant on the input data. It is a quantity which stays constant for all input data and varies between different tasks. It can therefore be described as task-dependant uncertainty.

![](../pictures/Homoscedasticity.png)

__Heteroscedastic uncertainty__
- For every observation (x,y) we have a different variance 
- A collection of random variables is heteroscedastic if there are sub-populations that have different variance from others.

_Example:_
- Measurement of heart rate 
- In the mornigng the heart rate ranging from 75 to 85 bpm after getting up from bed
- After riding the bike the measurements could vary from 120 to 160 bpm, depending how fast the person cycled

![](../pictures/Heteroscedasticity.png)

### Measurement uncertainty
- Another source of uncertainty is the measurement itself. 
- When the measurement is noisy, the uncertainty increases (Model’s confidence can be impaired if some of the pictures are taken using a bad quality camera)


Quelle: https://engineering.taboola.com/using-uncertainty-interpret-model/

## 3. Approaches in NN
### Dropout
### Bayes by Backpropagation
###

