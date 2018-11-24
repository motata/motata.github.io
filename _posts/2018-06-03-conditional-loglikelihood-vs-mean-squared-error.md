---
layout: post
title: Conditional Log-Likelihood vs. Mean Squared Error
comments: true
description: >
  This post introduces conditional log-likelihood and mean squared error and shows the relationship between these two performance measures.
---

For many machine learning problems, especially for supervised learning, the goal is to build a system that can take a vector $$\bold{x}\in{\mathbb{R}^m}$$ as input and predict the value of a scalar $$y\in{\mathbb{R}}$$ as its output. To solve such problems, we can follow the steps:
1. choose a model to predict a $$\hat{y}$$ when given an input $$\bold{x}$$
2. estimate the weights $$\theta = g(\bold{x})$$ that $$\hat{y} = f(\theta, \bold{x})$$
    1. define the performance measure (also known as loss function or cost function)
    2. design an algorithm that will improve the weights $$\theta$$ in a way that reduces loss function

For deep learning problems, step 1 corresponds to determining the architecture of the neural network, step 2.1 determining the cost function and step 2.2 training a model.

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
\theta_{ML} = \arg\max_\theta P(Y|X;\theta)
\end{aligned}
$$, where Y is all observed targets

If the examples are assumed to be i.i.d.(independently identically distributed), then this can be decomposed into 

$$
\begin{aligned}
  \theta_{ML} &= \arg\max_\theta\prod^n_{i=1}P(y^{(i)}|\bold{x}^{(i)};\theta)\\
              &= \arg\max_\theta\sum_{i=1}^n \log P(y^{(i)}|x^{(i)};\theta)
\end{aligned}
$$ 

Now we add the assumption of Gaussian distribution

$$ p(y|\bold{x}) = \mathcal{N}(y;\hat{y},\sigma^2) $$

Then the sum of log term will become:

$$
\begin{aligned}
\sum_{i=1}^n \log p(y^{(i)}|x^{(i)};\theta) = -n\log \sigma-\frac{n}{2}\log (2\pi)-\sum_{i=1}^n \frac{\lVert \hat{y}^{(i)}-y^{(i)}\rVert^2} {2\sigma^2}
\end{aligned}
$$

Comparing the log-likelihood with the mean squared error, we immediately see maximizing the log-likelihood with respect to $$\theta$$ yields the same estimate of the parameters $$\theta$$ as does minimizing the mean squared error.

**KL divergence**

Another way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $$p_{data}$$, defined by the training set and the model distribution $$\hat{p}_{model}$$.

In order to measure the dissimilarity of two probability distributions, we introduce KL divergence(Kullback-Leibler divergence), which is defined as follows:

$$
\begin{aligned}
\it{KL}(p\|q)\triangleq\sum_{k=1}^K p_k \log \frac{p_k}{q_k}
\end{aligned}
$$

If put the training set and model distribution in the KL function, we will get:

$$
\begin{aligned}
D_{KL}(p_{data}\|\hat{p}_model) = \Bbb{E}_{\bold{x}\sim p_{data}}[\log p_{data}(\bold{x}) - \log \hat{p}_{model}(\bold(x))]
\end{aligned}
$$

The term on the left doesn't depend on the model. This means when we train the model to minimize the KL divergence, we need only to minimize

$$
\begin{aligned}
-\Bbb{E}_{\bold{x}\sim p_{data}}[\log \hat{p}_{model}(\bold{x})]
\end{aligned}
$$

Now let's look bakc to the MLE. Because the arg max does not change when we rescale the cost function, we can divide the MLE function by n to obtain a version o f the criterion that is expressed as an expectation with respect to the empirical distribution $$p_{data}$$ defined by the training data:

$$
\begin{aligned}
\theta_{ML} = \arg\max_{\theta}\Bbb{E}_{\bold{x}\sim p_{data}}\log \hat{p}_{model}({\bold{x};\theta}),
\end{aligned}
$$

which is the same as the MLE result.




