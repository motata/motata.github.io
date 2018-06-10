Conditional Log-Likelihood vs. Mean Squared Error
=====

For many machine learning problems, especially for supervised learning, the goal is to build a system that can take a vector $$\bold{x}\in{\mathbb{R}^m}$$ as input and predict the value of a scalar $$y\in{\mathbb{R}}$$ as its output. To solve such problems, we can follow the steps:
1. choose a model to predict a $$\hat{y}$$ when given an input $$\bold{x}$$
2. estimate the weights theta of the model $$\theta = g(\bold{x})$$
    1. define the performance measure (also known as loss function or cost function)
    2. design an algorithm that will improve the weights $$\theta$$ in a way that reduces loss function

For deep learning problems, step 1 then corresponds to determining the architecture of the neural network, step 2.1 determining the cost function and step 2.2 training a model.

Conditional log-likelihood and mean squared error appear in the step 2.1. We can view them as different performance measures. But later we will see that with the assumption of a gaussian distribution of the training data, they are just different interpretations of the same performance measure.

## Mean Squared Error
Mean squared error (MSE) measures the euclidean distance between estimated targets and the real targets

$$
\begin{aligned}
MSE_{ML} = \dfrac{1}{m}\begin{Vmatrix}\text{\^{y}}-y\end{Vmatrix}_2^2
\end{aligned}
$$

## Conditional Log-Likelihood
Consider a set of n examples $$X = \{\bold{x}^{(1)}...\bold{x}^{(n)}\}$$ drawn independently from the true but unknown data-generating distribution $$p_{data}(\bold{x})$$. The conditional log-likelihood estimator is:

$$
\begin{aligned}
\theta_{ML} = \arg\max_\theta P(Y|X;\theta)\text{,}\quad\text{\small{where Y is all observed targets}}
\end{aligned}
$$

If the examples are assumed to be i.i.d.(independently identically distributed), then this can be decomposed into 

$$
\begin{aligned}
  \theta_{ML} &= \arg\max_\theta\prod^n_{i=1}P(y^{(i)}|\bold{x}^{(i)};\theta)\\
              &= \arg\max_\theta\sum_{i=1}^n \log P(y^{(i)}|x^{(i)};\theta)
\end{aligned}
$$

Now we add the assumption of Gaussian distribution $$ p(y|\bold{x})=\mathcal{N}(y;\hat{y},\sigma^2) $$:

$$
\begin{aligned}
\sum_{i=1}^n \log p(y^{(i)}|x^{(i)};\theta) = -n\log \sigma-\frac{n}{2}\log (2\pi)-\sum_{i=1}^n \frac{\lVert \hat{y}^{(i)}-y^{(i)}\rVert^2} {2\sigma^2}
\end{aligned}
$$

Comparing the log-likelihood with the mean squared error, we immediately see maximizing the log-likelihood with respect towyieldsthe same estimate of the parameterswas does minimizing the mean squared error.

**KL divergence**



