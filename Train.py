import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from metrics import pixel_accuracy, mean_iou, dice_coefficient, precision_recall_f1, iou_per_class
from losses import TverskyLoss

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, save_dir="checkpoints", log_dir="logs"):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir)

        os.makedirs(save_dir, exist_ok=True)

    def save_weights(self, epoch, filename="model_checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        print(f"Checkpoint saved at epoch {epoch}.")

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch

    def train(self, train_loader, val_loader=None, epochs=10, checkpoint_interval=5):

        self.model.to(self.device)
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            correct_predictions = 0
            total_pixels = 0

            for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == masks).sum().item()
                total_pixels += masks.numel()

            train_loss /= len(train_loader)
            train_accuracy = correct_predictions / total_pixels

            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

            if val_loader:
                val_loss, val_accuracy = self.validate(val_loader)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

            self.scheduler.step()

            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_weights(epoch, filename=f"model_epoch_{epoch+1}.pth")

    def validate(self, val_loader):

        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_pixels = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                # Metrics calculation
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == masks).sum().item()
                total_pixels += masks.numel()

        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_pixels

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy
