import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from datetime import datetime
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50
from torch.utils.data import DataLoader

class HatefulMemesModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super().__init__()
        
        # Image encoder (ResNet-50)
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, 512)
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(bert_model)
        self.text_projection = nn.Linear(768, 512)  # BERT hidden size to common space
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # 512 (image) + 512 (text) = 1024
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Binary classification
        )
        
    def forward(self, images, input_ids, attention_mask):
        # Image encoding
        image_features = self.image_encoder(images)
        
        # Text encoding
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
        
        # Concatenate features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Classification
        output = self.fusion(combined_features)
        return output.squeeze()

class KaggleHatefulMemesEvaluator:
    def __init__(self, model, device, log_dir='/kaggle/working/runs/hateful_memes'):
        """
        Initialize the evaluator for Kaggle environment
        
        Args:
            model: The model to evaluate
            device: torch.device for computation
            log_dir: Directory for TensorBoard logs (default: Kaggle working directory)
        """
        self.model = model
        self.device = device
        
        # Create timestamp-based directory to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"{log_dir}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize best metrics
        self.best_auroc = 0
        self.best_f1 = 0
        
    def evaluate(self, dataloader, epoch=0, mode='val'):
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Log to TensorBoard and save visualizations
        self.log_metrics(metrics, epoch, mode)
        self.generate_visualizations(metrics, epoch, mode)
        
        # Save best model if applicable
        if mode == 'val':
            self.save_best_model(metrics, epoch)
        
        return metrics
    
    def calculate_metrics(self, labels, preds, probs):
        """Calculate evaluation metrics"""
        auroc = roc_auc_score(labels, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, 
                                                                 average='binary')
        cm = confusion_matrix(labels, preds)
        
        return {
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def log_metrics(self, metrics, epoch, mode):
        """Log metrics to TensorBoard"""
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                self.writer.add_scalar(f'{mode}/{metric_name}', value, epoch)
                
        # Save metrics to CSV for Kaggle
        metrics_file = os.path.join(self.log_dir, f'{mode}_metrics.csv')
        with open(metrics_file, 'a') as f:
            if epoch == 0:
                f.write('epoch,auroc,precision,recall,f1\n')
            f.write(f"{epoch},{metrics['auroc']:.4f},{metrics['precision']:.4f},"
                   f"{metrics['recall']:.4f},{metrics['f1']:.4f}\n")
    
    def generate_visualizations(self, metrics, epoch, mode):
        """Generate and save visualizations"""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                   xticklabels=['Non-Hateful', 'Hateful'],
                   yticklabels=['Non-Hateful', 'Hateful'])
        plt.title(f'Confusion Matrix - {mode.capitalize()} (Epoch {epoch})')
        
        # Save plot to Kaggle working directory
        plt.savefig(os.path.join(self.log_dir, f'{mode}_confusion_matrix_epoch_{epoch}.png'))
        plt.close()
        
        # Add to TensorBoard
        self.writer.add_figure(f'{mode}/Confusion_Matrix', plt.gcf(), epoch)
    
    def save_best_model(self, metrics, epoch):
        """Save best model based on AUROC and F1 score"""
        if metrics['auroc'] > self.best_auroc:
            self.best_auroc = metrics['auroc']
            torch.save(self.model.state_dict(), 
                      os.path.join(self.log_dir, 'best_model_auroc.pth'))
            
        if metrics['f1'] > self.best_f1:
            self.best_f1 = metrics['f1']
            torch.save(self.model.state_dict(), 
                      os.path.join(self.log_dir, 'best_model_f1.pth'))
    
    def log_sample_predictions(self, batch, outputs, epoch, mode):
        """Log sample predictions with images and text"""
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # Create visualization grid
        num_samples = min(8, len(batch['image']))
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for idx in range(num_samples):
            img = batch['image'][idx].cpu().permute(1, 2, 0)
            img = torch.clamp(img * torch.tensor([0.229, 0.224, 0.225]) + 
                            torch.tensor([0.485, 0.456, 0.406]), 0, 1)
            
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            title = f"Text: {batch['text'][idx]}\n"
            title += f"True: {'Hateful' if batch['label'][idx] else 'Non-Hateful'}\n"
            title += f"Pred: {probs[idx].item():.2f}"
            
            axes[idx].set_title(title, fontsize=8)
        
        plt.tight_layout()
        
        # Save to Kaggle working directory
        plt.savefig(os.path.join(self.log_dir, f'{mode}_samples_epoch_{epoch}.png'))
        self.writer.add_figure(f'{mode}/Sample_Predictions', fig, epoch)
        plt.close()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

# Just start using the classes and functions defined above
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create datasets
train_dataset = HatefulMemesDataset(
    data_dir='/kaggle/input/facebook-hateful-meme-dataset/data',
    split='train',
    augment=True
)

val_dataset = HatefulMemesDataset(
    data_dir='/kaggle/input/facebook-hateful-meme-dataset/data',
    split='dev',  # Using dev set for validation
    augment=False
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_dataset.get_sampler(),
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Initialize model
model = HatefulMemesModel()
model = model.to(device)

# Initialize optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Initialize evaluator
evaluator = KaggleHatefulMemesEvaluator(model, device)

# Training loop
num_epochs = 10
best_val_auroc = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    avg_train_loss = train_loss / len(train_loader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}')
    
    # Validation phase
    val_metrics = evaluator.evaluate(val_loader, epoch, mode='val')
    print(f"Validation Metrics:")
    print(f"AUROC: {val_metrics['auroc']:.3f}")
    print(f"F1: {val_metrics['f1']:.3f}")
    
    # Log sample predictions
    batch = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(batch['image'].to(device),
                      batch['input_ids'].to(device),
                      batch['attention_mask'].to(device))
        evaluator.log_sample_predictions(batch, outputs, epoch, 'val')
    
    # Save best model
    if val_metrics['auroc'] > best_val_auroc:
        best_val_auroc = val_metrics['auroc']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auroc': best_val_auroc,
        }, f'/kaggle/working/best_model.pth')

# Final evaluation
print("\nTraining completed! Loading best model for final evaluation...")
checkpoint = torch.load('/kaggle/working/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = HatefulMemesDataset(
    data_dir='/kaggle/input/facebook-hateful-meme-dataset/data',
    split='test',
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

test_metrics = evaluator.evaluate(test_loader, epoch='final', mode='test')
print("\nFinal Test Metrics:")
print(f"AUROC: {test_metrics['auroc']:.3f}")
print(f"F1: {test_metrics['f1']:.3f}")

evaluator.close() 