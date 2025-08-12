# Deep State-Space Model with Neural Categorical Latents

This repository contains a PyTorch implementation of a Deep State-Space Model (DSSM) for unsupervised discovery of discrete states in sequential data. The model is based on the principles of variational inference, utilizing a structured posterior and the Gumbel-Softmax reparameterization trick to enable end-to-end training.

The implementation is designed to be clear and modular, using Hydra for configuration and Weights & Biases for experiment tracking. It includes a synthetic data generator that creates time-series data conforming to the Hidden Markov Model (HMM) assumptions, which is used to train the model and verify the correctness of the implementation.

-----

## 1\. Mathematical Formulation

We model a sequence of observations $X \\triangleq (x\_1, \\dots, x\_T)$ by assuming they are generated from a sequence of discrete latent states $C \\triangleq (c\_1, \\dots, c\_T)$, where each state $c\_t$ is a categorical variable, $c\_t \\in {1, \\dots, K}$.

### 1.1. The Generative Model

The generative process $p\_{\\theta}(X, C)$ defines the joint probability over observations and latent states. It is a Hidden Markov Model with neural network components for transitions and emissions.

$$p_{\theta}(X, C) = p(c_1) \prod_{t=2}^{T} p_{\theta}(c_t | c_{t-1}) \prod_{t=1}^{T} p_{\theta}(x_t | c_t)$$

This model consists of:

1.  **Initial State Distribution $p(c\_1)$**: A uniform categorical distribution over $K$ states.
2.  **Neural Transition Model $p\_{\\theta}(c\_t | c\_{t-1})$**: A neural network that takes the previous state $c\_{t-1}$ and outputs the probability distribution for the current state $c\_t$. This allows the model to learn complex, non-linear dynamics between states.
3.  **Emission Model $p\_{\\theta}(x\_t | c\_t)$**: A neural network that takes the current state $c\_t$ and outputs the parameters for the probability distribution of the observation $x\_t$. For this implementation, it outputs the mean of a Gaussian distribution with fixed variance.

### 1.2. The Structured Inference Model

Since the true posterior $p\_{\\theta}(C|X)$ is intractable, we introduce a variational distribution $q\_{\\phi}(C|X)$ to approximate it. To capture the temporal dependencies inherent in the data, we use a structured factorization:

$$q_{\phi}(C|X) = q_{\phi}(c_1 | X) \prod_{t=2}^{T} q_{\phi}(c_t | c_{t-1}, X)$$

This is implemented using Recurrent Neural Networks (RNNs). A Bi-directional RNN first processes the entire input sequence $X$ to produce context-aware hidden states $h\_t$ for each timestep. Then, a second, unidirectional RNN uses the context $h\_t$ and the previous state $c\_{t-1}$ to produce the parameters for the categorical distribution over the current state $c\_t$.

### 1.3. The Evidence Lower Bound (ELBO)

We train the model by maximizing the Evidence Lower Bound (ELBO), $\\mathcal{L}(\\theta, \\phi)$, which is a lower bound on the log-likelihood of the data, $\\log p\_{\\theta}(X)$.

The derivation starts with the log-likelihood:

$$\log p_{\theta}(X) = \log \int p_{\theta}(X, C) dC$$

We introduce the inference model $q\_{\\phi}(C|X)$:

$$\log p_{\theta}(X) = \log \int p_{\theta}(X, C) \frac{q_{\phi}(C|X)}{q_{\phi}(C|X)} dC = \log \mathbb{E}_{q_{\phi}(C|X)}\left[\frac{p_{\theta}(X, C)}{q_{\phi}(C|X)}\right]$$

By Jensen's inequality, we get the ELBO:

$$\log p_{\theta}(X) \ge \mathbb{E}_{q_{\phi}(C|X)}\left[\log \frac{p_{\theta}(X, C)}{q_{\phi}(C|X)}\right] \triangleq \mathcal{L}(\theta, \phi)$$

Expanding the ELBO gives:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_{\phi}(C|X)}[\log p_{\theta}(X, C)] - \mathbb{E}_{q_{\phi}(C|X)}[\log q_{\phi}(C|X)]$$

Substituting our factorized models and simplifying leads to the final objective function:

$$\mathcal{L} = \sum_{t=1}^{T} \mathbb{E}_{q_{\phi}(c_t|X)}[\log p_{\theta}(x_t | c_t)] - D_{KL}(q_{\phi}(c_1|X) \,\|\, p(c_1)) - \sum_{t=2}^{T} \mathbb{E}_{q_{\phi}(c_{t-1}|X)} \left[ D_{KL}(q_{\phi}(c_t | c_{t-1}, X) \,\|\, p_{\theta}(c_t | c_{t-1})) \right]$$

The objective is composed of two main parts:

1.  **Reconstruction Term**: $\\mathbb{E}*{q*{\\phi}(c\_t|X)}[\\log p\_{\\theta}(x\_t | c\_t)]$. This encourages the model to find latent states that can accurately reconstruct the observed data.

2.  **KL Divergence Term**: This regularizer pushes the approximate posterior $q\_{\\phi}$ to be close to the prior $p\_{\\theta}$. A key advantage of using categorical latents is that the KL divergence between two categorical distributions, $D\_{KL}(\\text{Categorical} ,|, \\text{Categorical})$, has an analytic form: $D\_{KL}(q ,|, p) = \\sum\_{k=1}^{K} q\_k \\log\\frac{q\_k}{p\_k}$. This allows for exact, low-variance gradient estimation for this part of the loss.

### 1.4. The Gumbel-Softmax Trick

The reconstruction term requires sampling a discrete state $c\_t$ from $q\_{\\phi}(c\_t|X)$, which is a non-differentiable operation. To overcome this, we use the Gumbel-Softmax trick. To sample from a categorical distribution with class probabilities $\\pi = (\\pi\_1, \\dots, \\pi\_K)$, we compute a continuous, differentiable "soft" sample vector $y = (y\_1, \\dots, y\_K)$ as follows:

$$y_k = \frac{\exp((\log(\pi_k) + g_k) / \tau)}{\sum_{j=1}^K \exp((\log(\pi_j) + g_j) / \tau)}$$

where $g\_k \\sim \\text{Gumbel}(0, 1)$ are i.i.d. samples and $\\tau$ is a temperature parameter. As $\\tau \\to 0$, the vector $y$ approaches a one-hot encoding. During training, we anneal $\\tau$ from a high value to a low value, allowing gradients to flow through the sampling step.

-----

## 2\. Implementation Details (`model.py`)

This section maps the mathematical terms to their corresponding implementation in `dssm/model.py`.

  * **Inference Model $q\_{\\phi}(C|X)$**:

      * **Context Generation**: The Bi-directional GRU `self.birnn` takes the input sequence `x` and produces the context vectors `h_birnn`.
      * **Structured Posterior $q\_{\\phi}(c\_t | c\_{t-1}, X)$**: This is implemented by the unidirectional `self.inference_rnn` and the fully-connected layer `self.fc_inference`. At each timestep `t`, the `inference_rnn` takes the concatenation of the context `h_birnn[:, t, :]` and the previous soft state `prev_c_soft` as input. The output is passed to `fc_inference` to produce the logits `q_logits_t` for the categorical distribution $q\_{\\phi}(c\_t|...)$.

  * **Generative Model $p\_{\\theta}(X, C)$**:

      * **Neural Transition Model $p\_{\\theta}(c\_t | c\_{t-1})$**: This is `self.transition_net`, a simple `nn.Sequential` MLP. It takes the previous soft state `prev_c_soft` as input and outputs the logits for the prior distribution $p\_{\\theta}(c\_t|c\_{t-1})$.
      * **Emission Model $p\_{\\theta}(x\_t | c\_t)$**: This is `self.emission_net`, another `nn.Sequential` MLP. It takes the Gumbel-Softmax sample `c_t_soft` as input and outputs the mean `emission_mean_t` of the Gaussian distribution for reconstructing the observation $x\_t$.

  * **Loss Calculation (in `forward` method)**:

      * **Reconstruction Loss**: The `emission_mean_t` from the emission net is used to define a `torch.distributions.Normal` distribution. The loss is the negative `log_prob` of the true data `x[:, t, :]` under this distribution, encouraging the model to maximize the probability of the observed data.
      * **KL Divergence**: The logits from the inference net (`q_logits_t`) and the transition net (`p_logits_t`) are used to create two `torch.distributions.Categorical` objects. The KL divergence is then computed analytically using `torch.distributions.kl.kl_divergence`, providing a stable, exact gradient.

-----

## 3\. Synthetic Data Generation (`data.py`)

The script `dssm/data.py` generates a synthetic dataset that is ideal for testing this model. It creates data from a true Hidden Markov Model process.

1.  **State Transition**: A transition matrix is created with a strong diagonal bias controlled by the `transition_bias` hyperparameter. A high value (e.g., 0.9) means that the process has a 90% chance of staying in the same state at each timestep. This creates temporally coherent sequences with clear state persistence.
2.  **State Emission**: Each of the $K$ latent states is assigned a unique, randomly generated mean vector in the feature space.
3.  **Observation Generation**: At each timestep, an observation is generated by taking the mean vector corresponding to the current true state and adding Gaussian noise.

This process yields a dataset where distinct, persistent hidden states generate the observed sequences, providing a clear and verifiable learning target for the DSSM. The dataloader provides both the observations `x` and the true latent states `c`, allowing for the calculation of clustering accuracy.

-----

## 4\. How to Use

### 4.1. Setup

First, install the required dependencies:

```bash
pip install torch numpy wandb hydra-core omegaconf scipy
```

### 4.2. Configuration

All hyperparameters for the data, model, and trainer are managed in `configs/config.yaml`. You can modify this file to experiment with different settings.

### 4.3. Training

To run the training, execute the main script from the project's root directory:

```bash
python dssm/main.py
```

Hydra allows you to override any configuration parameter from the command line. For example, to run on the CPU for 10 epochs:

```bash
python dssm/main.py trainer.device=cpu trainer.n_epochs=10
```

Training progress, losses, and accuracy metrics will be logged to your Weights & Biases project.

### 4.4. Evaluation and Accuracy

A critical challenge in evaluating unsupervised clustering models is **state permutation invariance**. The model may learn the correct state clusters but assign them arbitrary integer labels (e.g., its learned state '2' might correspond to the true state '0').

To address this, the `_calculate_accuracy` function in `dssm/main.py` does not perform a naive comparison. Instead, at the end of each validation epoch, it:

1.  Computes a confusion matrix between all predicted states and all true states for the entire validation set.
2.  Uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to find the optimal one-to-one mapping between predicted and true state labels that maximizes the diagonal of the confusion matrix.
3.  Calculates the accuracy based on this optimal mapping.

This provides a true measure of the model's ability to identify the latent states, regardless of the labels it assigns to them.
