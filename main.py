import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

from data import get_dataloader

# Import your model and data loader
from model import DSSM


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.trainer.device if torch.cuda.is_available() else "cpu",
        )
        print(f"Using device: {self.device}")

        # Initialize wandb
        wandb.init(
            project=cfg.trainer.wandb_project,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

        # Data
        self.train_loader, self.val_loader = get_dataloader(cfg)

        # Model
        self.model = DSSM(cfg).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.trainer.learning_rate,
        )

        self.tau = cfg.trainer.gumbel_tau

    def _calculate_accuracy(self, all_q_probs, all_true_states):
        """
        Calculates accuracy after finding the optimal mapping between predicted
        and true states using the Hungarian algorithm. This corrects for the
        permutation invariance problem in unsupervised clustering.

        Args:
            all_q_probs (Tensor): Concatenated q_probs for the entire dataset.
                                  Shape: (num_samples, seq_len, n_states)
            all_true_states (Tensor): Concatenated true states for the entire dataset.
                                      Shape: (num_samples, seq_len)

        Returns:
            float: The accuracy score, corrected for permutation.
        """
        with torch.no_grad():
            # Get hard predictions
            all_predicted_states = (
                torch.argmax(all_q_probs, dim=2).cpu().numpy()
            )
            all_true_states = all_true_states.cpu().numpy()

            n_states = self.cfg.model.n_states

            # Create a confusion matrix (cost matrix)
            confusion_matrix = np.zeros((n_states, n_states), dtype=np.int64)
            for true_label, pred_label in zip(
                all_true_states.flatten(),
                all_predicted_states.flatten(),
            ):
                confusion_matrix[pred_label, true_label] += 1

            # Use the Hungarian algorithm to find the optimal assignment
            # We want to maximize the diagonal of the confusion matrix, so we
            # pass the negative matrix to the algorithm (which minimizes cost).
            row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

            # The accuracy is the sum of the diagonal elements of the reordered
            # confusion matrix, divided by the total number of samples.
            correct_predictions = confusion_matrix[row_ind, col_ind].sum()
            accuracy = correct_predictions / all_true_states.size

            return accuracy

    def _run_epoch(self, epoch, data_loader, is_train=True):
        """Generic epoch runner for both training and validation."""
        if is_train:
            self.model.train()
            mode = "Train"
        else:
            self.model.eval()
            mode = "Val"

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        # Store all predictions and labels for the epoch
        epoch_q_probs = []
        epoch_true_states = []

        for data, true_states in data_loader:
            data = data.to(self.device)
            true_states = true_states.to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            loss, recon_loss, kl_loss, q_probs = self.model(data, self.tau)

            if is_train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            epoch_q_probs.append(q_probs.detach())
            epoch_true_states.append(true_states.detach())

        # After the epoch, calculate metrics
        avg_loss = total_loss / len(data_loader)
        avg_recon_loss = total_recon_loss / len(data_loader)
        avg_kl_loss = total_kl_loss / len(data_loader)

        # Concatenate all batch results for accurate metric calculation
        all_q_probs = torch.cat(epoch_q_probs, dim=0)
        all_true_states = torch.cat(epoch_true_states, dim=0)

        # Calculate accuracy with the corrected method
        accuracy = self._calculate_accuracy(all_q_probs, all_true_states)

        # Logging
        log_data = {
            f"{mode.lower()}/epoch": epoch,
            f"{mode.lower()}/loss": avg_loss,
            f"{mode.lower()}/reconstruction_loss": avg_recon_loss,
            f"{mode.lower()}/kl_loss": avg_kl_loss,
            f"{mode.lower()}/accuracy": accuracy,
        }
        if is_train:
            log_data["train/tau"] = self.tau

        wandb.log(log_data)
        print(
            f"Epoch {epoch} [{mode}]: Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Acc: {accuracy:.4f}",
        )

    def run(self):
        print("Starting training...")
        for epoch in range(1, self.cfg.trainer.n_epochs + 1):
            self._run_epoch(epoch, self.train_loader, is_train=True)
            with torch.no_grad():
                self._run_epoch(epoch, self.val_loader, is_train=False)

            # Anneal Gumbel-Softmax temperature
            self.tau = max(
                self.cfg.trainer.min_tau,
                self.tau * self.cfg.trainer.tau_decay,
            )

        print("Training finished.")
        wandb.finish()


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
