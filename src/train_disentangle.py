import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import os
import time
import logging
import argparse
from tqdm import tqdm
from typing import Dict, Optional

from .Flare_Disentangle import FlareDisentanglementNetwork
from .utils.loss import FlareDisentanglementLoss
from .utils.checkpoint import (
    load_disentanglement_checkpoint,
    save_disentanglement_checkpoint,
)
from dataloader.paired_datasets import FlareDisentanglementDataset

class FlareDisentanglementTrainer:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        model_config = {
            'in_channels': config.get('in_channels', 3),
            'base_channels': config.get('base_channels', 64)
        }
        self.model = FlareDisentanglementNetwork(**model_config)
        self.model.to(self.device)

        self.criterion = FlareDisentanglementLoss(
            lambda_supervised=config.get('lambda_supervised', 1.0),
            lambda_physical=config.get('lambda_physical', 0.5),
            lambda_orthogonal=config.get('lambda_orthogonal', 0.2)
        )
        self.criterion.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200),
            eta_min=config.get('min_lr', 1e-7)
        )

        self.epochs = config.get('epochs', 200)
        self.save_interval = config.get('save_interval', 20)
        self.log_interval = config.get('log_interval', 10)
        self.val_interval = config.get('val_interval', 5)

        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.tensorboard_dir = config.get('tensorboard_dir') or os.path.join(
            self.checkpoint_dir,
            'tensorboard',
        )
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        self.setup_logging()

        config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
        self.writer.add_text('config', config_text, 0)

        self.logger.info(f"TensorBoard logs: {self.tensorboard_dir}")

    def setup_logging(self):
        log_file = os.path.join(self.checkpoint_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_dataloaders(self):
        train_dataset = FlareDisentanglementDataset(
            dataset_config_path=self.config['dataset_config_path'],
            height=self.config.get('image_height', 512),
            width=self.config.get('image_width', 512),
            flare_config_path=self.config.get('flare_config_path', 'dataloader/flare_config.yml')
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        if self.config.get('val_dataset_config_path'):
            val_dataset = FlareDisentanglementDataset(
                dataset_config_path=self.config['val_dataset_config_path'],
                height=self.config.get('image_height', 512),
                width=self.config.get('image_width', 512),
                flare_config_path=self.config.get('flare_config_path', 'dataloader/flare_config.yml')
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('val_batch_size', 4),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        else:
            self.val_loader = None

        self.logger.info(f"Training samples: {len(train_dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(val_dataset)}")

    def train_epoch(self, epoch: int):
        self.model.train()

        epoch_losses = {}
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                lq_image = batch['input_image'].to(self.device)

                flare_gt = batch.get('flare_gt')
                lol_gt = batch.get('lol_gt')

                if flare_gt is not None:
                    flare_gt = flare_gt.to(self.device)
                if lol_gt is not None:
                    lol_gt = lol_gt.to(self.device)

                self.optimizer.zero_grad()

                _, background, flare, orth_loss = self.model(lq_image)

                total_loss, loss_dict = self.criterion(
                    background, flare, lq_image, orth_loss,
                    flare_gt=flare_gt, lol_gt=lol_gt
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 1.0)
                )

                self.optimizer.step()

                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if key not in epoch_losses:
                            epoch_losses[key] = []
                        epoch_losses[key].append(value.item())

                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Recon': f"{loss_dict.get('physical_reconstruction', 0):.4f}",
                    'Orth': f"{orth_loss.item():.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

                if batch_idx % self.log_interval == 0:
                    self.log_batch_metrics(epoch, batch_idx, loss_dict, num_batches)

        avg_losses = {}
        for key, values in epoch_losses.items():
            avg_losses[key] = sum(values) / len(values)

        return avg_losses

    def validate(self, epoch: int):
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {}
        reconstruction_errors = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                lq_image = batch['input_image'].to(self.device)

                flare_gt = batch.get('flare_gt')
                lol_gt = batch.get('lol_gt')

                if flare_gt is not None:
                    flare_gt = flare_gt.to(self.device)
                if lol_gt is not None:
                    lol_gt = lol_gt.to(self.device)

                _, background, flare, orth_loss = self.model(lq_image)

                total_loss, loss_dict = self.criterion(
                    background, flare, lq_image, orth_loss,
                    flare_gt=flare_gt, lol_gt=lol_gt
                )

                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        if key not in val_losses:
                            val_losses[key] = []
                        val_losses[key].append(value.item())

                reconstructed = background + flare
                recon_error = torch.mean((reconstructed - lq_image) ** 2)
                reconstruction_errors.append(recon_error.item())

        avg_val_losses = {}
        for key, values in val_losses.items():
            avg_val_losses[key] = sum(values) / len(values)

        avg_val_losses['avg_reconstruction_error'] = sum(reconstruction_errors) / len(reconstruction_errors)
        return avg_val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        save_disentanglement_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, epoch, self.config
        )

        if (epoch + 1) % self.save_interval == 0:
            numbered_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch+1:03d}.pth')
            save_disentanglement_checkpoint(
                numbered_path, self.model, self.optimizer, self.scheduler, epoch, self.config
            )

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            save_disentanglement_checkpoint(
                best_path, self.model, self.optimizer, self.scheduler, epoch, self.config
            )
            self.logger.info(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = load_disentanglement_checkpoint(
            self.model, self.optimizer, self.scheduler, checkpoint_path, map_location=self.device
        )

        epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Resumed from checkpoint at epoch {epoch}")

        return epoch

    def log_batch_metrics(self, epoch: int, batch_idx: int, loss_dict: Dict, num_batches: int):
        step = epoch * num_batches + batch_idx

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'train/batch_{key}', value.item(), step)

        self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], step)

        loss_str = ' | '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()
                              if isinstance(v, torch.Tensor)])
        self.logger.info(f'Epoch {epoch+1:3d} | Batch {batch_idx:4d}/{num_batches:4d} | {loss_str}')

    def log_epoch_metrics(self, epoch: int, train_losses: Dict, val_losses: Dict):
        train_str = ' | '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()])
        self.logger.info(f'Epoch {epoch+1:3d} Train | {train_str}')

        if val_losses:
            val_str = ' | '.join([f'{k}: {v:.4f}' for k, v in val_losses.items()])
            self.logger.info(f'Epoch {epoch+1:3d} Val   | {val_str}')

        for key, value in train_losses.items():
            self.writer.add_scalar(f'train_epoch/{key}', value, epoch + 1)

        for key, value in val_losses.items():
            self.writer.add_scalar(f'val_epoch/{key}', value, epoch + 1)

        self.writer.flush()

    def save_sample_results(self, epoch: int, num_samples=4):
        if self.val_loader is None:
            return

        self.model.eval()
        save_dir = os.path.join(self.checkpoint_dir, 'samples')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break

                lq_image = batch['input_image'].to(self.device)
                lol_gt = batch.get('lol_gt')
                flare_gt = batch.get('flare_gt')

                if lol_gt is not None:
                    lol_gt = lol_gt.to(self.device)
                if flare_gt is not None:
                    flare_gt = flare_gt.to(self.device)

                _, background, flare, _ = self.model(lq_image)

                reconstructed = background + flare

                sample_images = [lq_image[0:1]]
                sample_labels = ['Input']

                if lol_gt is not None:
                    sample_images.append(lol_gt[0:1])
                    sample_labels.append('LOL_GT')

                if flare_gt is not None:
                    sample_images.append(flare_gt[0:1])
                    sample_labels.append('Flare_GT')

                sample_images.extend([
                    background[0:1],
                    flare[0:1],
                    reconstructed[0:1]
                ])
                sample_labels.extend(['Background', 'Flare', 'Reconstructed'])

                sample_grid = torch.cat(sample_images, dim=0)
                grid = vutils.make_grid(sample_grid, nrow=len(sample_images), normalize=True, padding=2)

                self.writer.add_image(f'samples/epoch_{epoch}_sample_{i}', grid, epoch)

                sample_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i}.png')
                vutils.save_image(grid, sample_path)

                for j, (img, label) in enumerate(zip(sample_images, sample_labels)):
                    individual_img = vutils.make_grid(img, normalize=True, padding=2)
                    self.writer.add_image(f'components/{label}/epoch_{epoch}_sample_{i}', individual_img, epoch)

        self.logger.info(f"Saved sample results to {save_dir} and TensorBoard")

    def save_sample_results_step(self, step: int, num_samples=2):
        if self.val_loader is None:
            return

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break

                lq_image = batch['input_image'].to(self.device)
                lol_gt = batch.get('lol_gt')
                flare_gt = batch.get('flare_gt')

                if lol_gt is not None:
                    lol_gt = lol_gt.to(self.device)
                if flare_gt is not None:
                    flare_gt = flare_gt.to(self.device)

                _, background, flare, _ = self.model(lq_image)

                reconstructed = background + flare

                sample_images = [lq_image[0:1]]
                sample_labels = ['Input']

                if lol_gt is not None:
                    sample_images.append(lol_gt[0:1])
                    sample_labels.append('LOL_GT')

                if flare_gt is not None:
                    sample_images.append(flare_gt[0:1])
                    sample_labels.append('Flare_GT')

                sample_images.extend([
                    background[0:1],
                    flare[0:1],
                    reconstructed[0:1]
                ])
                sample_labels.extend(['Background', 'Flare', 'Reconstructed'])

                sample_grid = torch.cat(sample_images, dim=0)
                grid = vutils.make_grid(sample_grid, nrow=len(sample_images), normalize=True, padding=2)

                self.writer.add_image(f'step_samples/step_{step}_sample_{i}', grid, step)

                for j, (img, label) in enumerate(zip(sample_images, sample_labels)):
                    individual_img = vutils.make_grid(img, normalize=True, padding=2)
                    self.writer.add_image(f'step_components/{label}/step_{step}_sample_{i}', individual_img, step)

        self.model.train()

    def add_model_graph(self):
        try:
            dummy_input = torch.randn(1, 3,
                                    self.config.get('image_height', 512),
                                    self.config.get('image_width', 512)).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
            self.logger.info("Added model graph to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Could not add model graph to TensorBoard: {e}")

    def train(self):
        self.logger.info("Starting training")
        self.logger.info(f"Config: {self.config}")

        self.create_dataloaders()

        start_epoch = 0
        if self.config.get('resume_from'):
            start_epoch = self.load_checkpoint(self.config['resume_from'])

        best_val_loss = float('inf')

        for epoch in range(start_epoch, self.epochs):
            train_losses = self.train_epoch(epoch)

            val_losses = {}
            if epoch % self.val_interval == 0:
                val_losses = self.validate(epoch)
                if epoch % (self.val_interval * 2) == 0:
                    self.save_sample_results(epoch)

            self.scheduler.step()

            self.log_epoch_metrics(epoch, train_losses, val_losses)

            is_best = False
            if val_losses and 'total' in val_losses:
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    is_best = True
                    self.logger.info(f"New best validation loss: {best_val_loss:.6f}")

            self.save_checkpoint(epoch, is_best)

        self.logger.info("Training complete")

        self.writer.close()
        self.logger.info(f"TensorBoard command: tensorboard --logdir {self.tensorboard_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='LUCID flare disentanglement training')

    parser.add_argument('--dataset_config_path', type=str, required=True,
                       help='Path to dataset configuration file (.yml)')
    parser.add_argument('--val_dataset_config_path', type=str,
                       help='Path to validation dataset configuration file (.yml, optional)')
    parser.add_argument('--flare_config_path', type=str, default='dataloader/flare_config.yml',
                       help='Path to flare configuration file')

    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    parser.add_argument('--lambda_supervised', type=float, default=1.0, help='Supervised loss weight')
    parser.add_argument('--lambda_physical', type=float, default=0.5, help='Physical constraint loss weight')
    parser.add_argument('--lambda_orthogonal', type=float, default=0.2, help='Orthogonal loss weight')

    parser.add_argument('--in_channels', type=int, default=3, help='Input channels')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels')
    parser.add_argument('--image_height', type=int, default=512, help='Image height')
    parser.add_argument('--image_width', type=int, default=512, help='Image width')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=20, help='Save checkpoint interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    parser.add_argument('--vis_interval', type=int, default=1500, help='Visualization interval in steps')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--tensorboard_dir', type=str, help='TensorBoard log directory (default: checkpoint_dir/tensorboard)')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    parser.add_argument('--resume_from', type=str, help='Resume training from checkpoint')

    return parser.parse_args()

def inference_single_image(model, image_path, device='cuda'):
    """Run single-image flare disentanglement inference."""
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    lq_image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        background_context, background, flare, _ = model(lq_image)

        reconstructed = background + flare

    return {
        'background_context': background_context,
        'background': background,
        'flare': flare,
        'reconstructed': reconstructed,
        'original': lq_image,
    }

if __name__ == "__main__":
    args = parse_args()

    config = vars(args)

    trainer = FlareDisentanglementTrainer(config)
    trainer.train()
