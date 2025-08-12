import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSM(nn.Module):
    """
    Deep State-Space Model with Neural Categorical Latent Variables.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_states = cfg.model.n_states
        self.n_features = cfg.model.n_features
        self.rnn_hidden_size = cfg.model.rnn_hidden_size
        self.embedding_dim = cfg.model.embedding_dim

        # --- Inference Model (q_phi) ---
        # Bi-RNN to get context from observations
        self.birnn = nn.GRU(
            self.n_features,
            self.rnn_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Unidirectional RNN for structured posterior
        self.inference_rnn = nn.GRU(
            self.rnn_hidden_size * 2 + self.n_states,
            self.rnn_hidden_size,
            batch_first=True,
        )
        self.fc_inference = nn.Linear(self.rnn_hidden_size, self.n_states)

        # --- Generative Model (p_theta) ---
        # State embeddings
        self.state_embedding = nn.Embedding(self.n_states, self.embedding_dim)

        # Neural Transition Model
        self.transition_net = nn.Sequential(
            nn.Linear(self.n_states, cfg.model.transition_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.model.transition_hidden_size, self.n_states),
        )

        # Emission Model
        self.emission_net = nn.Sequential(
            nn.Linear(self.n_states, cfg.model.emission_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.model.emission_hidden_size, self.n_features),
        )

    def forward(self, x, tau):
        """
        x: (batch_size, seq_len, n_features)
        tau: Gumbel-Softmax temperature

        Returns:
            elbo: The evidence lower bound loss.
            reconstruction_loss: The reconstruction component of the loss.
            kl_loss: The KL divergence component of the loss.
            q_probs: The predicted state probabilities from the inference network.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # --- Inference ---
        # Get context from Bi-RNN
        h_birnn, _ = self.birnn(x)  # (batch, seq_len, 2 * rnn_hidden)

        # Initialize for inference RNN loop
        q_probs_list = []
        kl_loss = 0
        reconstruction_loss = 0

        # Initial state for inference RNN
        prev_c_soft = torch.zeros(batch_size, self.n_states, device=device)
        inference_h = torch.zeros(
            1,
            batch_size,
            self.rnn_hidden_size,
            device=device,
        )

        # Initial prior (uniform)
        prior_c1_dist = torch.distributions.Categorical(
            logits=torch.zeros(batch_size, self.n_states, device=device),
        )

        for t in range(seq_len):
            # --- q(c_t | c_{t-1}, x) ---
            inference_input = torch.cat(
                [h_birnn[:, t, :], prev_c_soft],
                dim=1,
            ).unsqueeze(1)
            inference_out, inference_h = self.inference_rnn(
                inference_input,
                inference_h,
            )
            q_logits_t = self.fc_inference(inference_out.squeeze(1))
            q_dist_t = torch.distributions.Categorical(logits=q_logits_t)
            q_probs_list.append(q_dist_t.probs)

            # Sample c_t using Gumbel-Softmax
            c_t_soft = F.gumbel_softmax(
                q_logits_t,
                tau=tau,
                hard=False,
            )  # (batch, n_states)

            # --- p(c_t | c_{t-1}) ---
            if t == 0:
                p_dist_t = prior_c1_dist
            else:
                p_logits_t = self.transition_net(prev_c_soft)
                p_dist_t = torch.distributions.Categorical(logits=p_logits_t)

            # --- KL Divergence ---
            kl_div = torch.distributions.kl.kl_divergence(
                q_dist_t,
                p_dist_t,
            ).mean()
            kl_loss += kl_div

            # --- Reconstruction ---
            # p(x_t | c_t)
            emission_mean_t = self.emission_net(c_t_soft)
            # Assuming a Gaussian emission with fixed variance for simplicity
            recon_dist = torch.distributions.Normal(emission_mean_t, 1.0)
            log_prob_xt = recon_dist.log_prob(x[:, t, :]).sum(-1).mean()
            reconstruction_loss -= log_prob_xt  # We want to maximize log prob, so minimize negative

            # Update previous state for next timestep
            prev_c_soft = c_t_soft

        # Combine losses for ELBO
        elbo = reconstruction_loss + kl_loss

        # Stack probabilities for accuracy calculation
        q_probs = torch.stack(
            q_probs_list,
            dim=1,
        )  # (batch, seq_len, n_states)

        return elbo, reconstruction_loss, kl_loss, q_probs
