import hydra
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf

from data import get_dataloader
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

    def _calculate_accuracy(self, q_probs, true_states):
        """
        Calculates the accuracy of state predictions.
        q_probs: (batch, seq_len, n_states)
        true_states: (batch, seq_len)
        """
        # Get the predicted state for each timestep by taking the argmax
        predicted_states = torch.argmax(q_probs, dim=2)

        # Compare predicted states with true states and calculate the mean
        correct_predictions = (predicted_states == true_states).float().sum()
        accuracy = (
            correct_predictions / true_states.numel()
        )  # numel() gives total number of elements
        return accuracy.item()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_accuracy = 0

        for batch_idx, (data, true_states) in enumerate(self.train_loader):
            data = data.to(self.device)
            true_states = true_states.to(self.device)
            self.optimizer.zero_grad()

            # The model now returns q_probs
            loss, recon_loss, kl_loss, q_probs = self.model(data, self.tau)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_accuracy += self._calculate_accuracy(q_probs, true_states)

        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)

        wandb.log(
            {
                "train/epoch": epoch,
                "train/loss": avg_loss,
                "train/reconstruction_loss": avg_recon_loss,
                "train/kl_loss": avg_kl_loss,
                "train/accuracy": avg_accuracy,
                "train/tau": self.tau,
            },
        )
        print(
            f"Epoch {epoch} [Train]: Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Acc: {avg_accuracy:.4f}",
        )

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for data, true_states in self.val_loader:
                data = data.to(self.device)
                true_states = true_states.to(self.device)

                loss, recon_loss, kl_loss, q_probs = self.model(data, self.tau)

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_accuracy += self._calculate_accuracy(
                    q_probs,
                    true_states,
                )

        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)

        wandb.log(
            {
                "val/epoch": epoch,
                "val/loss": avg_loss,
                "val/reconstruction_loss": avg_recon_loss,
                "val/kl_loss": avg_kl_loss,
                "val/accuracy": avg_accuracy,
            },
        )
        print(
            f"Epoch {epoch} [Val]:   Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Acc: {avg_accuracy:.4f}",
        )

    def run(self):
        print("Starting training...")
        for epoch in range(1, self.cfg.trainer.n_epochs + 1):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

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

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    # To run this, you would execute from the command line:
    # python dssm/main.py
    # You can override parameters like this:
    # python dssm/main.py trainer.device=cpu trainer.n_epochs=10
    main()
