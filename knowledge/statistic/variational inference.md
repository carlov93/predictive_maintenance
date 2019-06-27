## Variational inference

In short, variational inference is akin to what happens at every presentation you've attended. Someone in the audience asks the presenter a very difficult answer which he/she can't answer. The presenter conveniently reframes the question in an easier manner and gives an exact answer to that reformulated question rather than answering the original difficult question. 

In many interesting statistical problems, we can't directly calculate the posterior because the normalization constant is intractable. This happens often in latent variable models. For example assume that X represents a set of observations and Z represents a set of latent variables. If we are interested in the posterior P(Z|X), we know that

𝑃(𝑍|𝑋)=𝑃(𝑍,𝑋)/∫𝑧𝑃(𝑍,𝑋)

but often times we can't calculate the denominator.
 
One popular approach is MCMC, where we can sample exactly from the true posterior distribution; however, convergence can be prohibitively slow if we have many parameters to sample. This is where variational inference comes in handy. Variational inference seeks to approximate the true posterior, P(Z|X), with an approximate variational distribution, which we can calculate more easily. For notation, let V be the parameters of the variational distribution.

𝑃(𝑍|𝑋)≈𝑄(𝑍|𝑉)=∏𝑖𝑄(𝑍𝑖|𝑉𝑖)

Typically, in the true posterior distribution, the latent variables are not independent given the data, but if we restrict our family of variational distributions to a distribution that factorizes over each variable in Z (this is called a mean field approximation), our problem becomes a lot easier. We can easily pick each V_i so that Q(Z|V) is as close to P(Z|X) as possible when measured by Kullback Leibler (KL) divergence. Thus, our problem of interest is now selecting a V* such that

𝑉⋆=argmin𝑉𝐾𝐿(𝑄(𝑍|𝑉)||𝑃(𝑍|𝑋)

When you write out the formula for KL divergence, you'll notice that we now have a sum of terms involving V, which we can minimize. So now our estimation procedure turns into an optimization problem.

Once we arrive at a V*, we can use Q(Z|V*) as our best guess at the posterior when performing estimation or inference.


variational inference --> converting an inference problem in a probabilistic model to an optimization problem