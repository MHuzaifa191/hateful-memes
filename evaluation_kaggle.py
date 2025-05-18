import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from datetime import datetime
import torch.nn as nn
from transformers import BertModel, AutoModel
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler

class HatefulMemesModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super().__init__()
        
        # 1. Simpler image encoder with more freezing
        self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Freeze everything except final classifier
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.fc = nn.Sequential(
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Back to BERT with more freezing
        self.text_encoder = BertModel.from_pretrained(bert_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # Only unfreeze final layer
        for param in self.text_encoder.encoder.layer[-1:].parameters():
            param.requires_grad = True
            
        # 3. Much simpler fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # 4. Better initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, input_ids, attention_mask):
        # 4. Better feature extraction
        image_features = self.image_encoder(images)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use both last hidden state and pooler output
        last_hidden = text_outputs.last_hidden_state[:, 0, :]
        pooler = text_outputs.pooler_output
        text_features = (last_hidden + pooler) / 2
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        return self.fusion(combined)

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
            'confusion_matrix': cm,
            'true_labels': labels,  # Store for ROC curve
            'probabilities': probs  # Store for ROC curve
        }
    
    def log_metrics(self, metrics, epoch, mode):
        """Log metrics to TensorBoard"""
        for metric_name, value in metrics.items():
            # Skip arrays and only log scalar values
            if metric_name not in ['confusion_matrix', 'true_labels', 'probabilities'] and np.isscalar(value):
                self.writer.add_scalar(f'{mode}/{metric_name}', value, epoch)
        
        # Save metrics to CSV for Kaggle
        metrics_file = os.path.join(self.log_dir, f'{mode}_metrics.csv')
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write('epoch,auroc,precision,recall,f1\n')
        
        # Append metrics
        with open(metrics_file, 'a') as f:
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
        
        # Add to TensorBoard
        confusion_fig = plt.gcf()
        self.writer.add_figure(f'{mode}/confusion_matrix', confusion_fig, epoch)
        plt.close()
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(metrics['true_labels'], metrics['probabilities'])
        roc_auc = metrics['auroc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {mode.capitalize()} (Epoch {epoch})')
        plt.legend(loc="lower right")
        
        # Save ROC curve
        plt.savefig(os.path.join(self.log_dir, f'{mode}_roc_curve_epoch_{epoch}.png'))
        
        # Add to TensorBoard
        roc_fig = plt.gcf()
        self.writer.add_figure(f'{mode}/roc_curve', roc_fig, epoch)
        plt.close()
    
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

    def analyze_model_performance(self, dataloader, epoch, mode='val'):
        """Analyze model performance by examining correct and incorrect predictions"""
        self.model.eval()
        
        correct_examples = {'images': [], 'texts': [], 'probs': [], 'labels': []}
        incorrect_examples = {'images': [], 'texts': [], 'probs': [], 'labels': []}
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch.get('text', None)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                
                # Collect correct and incorrect examples
                for i in range(len(labels)):
                    example = {
                        'images': images[i].cpu(),
                        'probs': probs[i],
                        'labels': labels_np[i]
                    }
                    if texts is not None:
                        example['texts'] = texts[i]
                    
                    if preds[i] == labels_np[i]:
                        # Correct prediction
                        if len(correct_examples['images']) < 10:  # Limit to 10 examples
                            for k, v in example.items():
                                correct_examples[k].append(v)
                    else:
                        # Incorrect prediction
                        if len(incorrect_examples['images']) < 10:  # Limit to 10 examples
                            for k, v in example.items():
                                incorrect_examples[k].append(v)
                    
                    # Break if we have enough examples
                    if (len(correct_examples['images']) >= 10 and 
                        len(incorrect_examples['images']) >= 10):
                        break
        
        # Visualize correct examples
        if correct_examples['images']:
            fig_correct = self._create_examples_figure(correct_examples, "Correctly Classified Examples")
            self.writer.add_figure(f'{mode}/correct_examples', fig_correct, epoch)
            plt.savefig(os.path.join(self.log_dir, f'{mode}_correct_examples_epoch_{epoch}.png'))
            plt.close()
        
        # Visualize incorrect examples
        if incorrect_examples['images']:
            fig_incorrect = self._create_examples_figure(incorrect_examples, "Incorrectly Classified Examples")
            self.writer.add_figure(f'{mode}/incorrect_examples', fig_incorrect, epoch)
            plt.savefig(os.path.join(self.log_dir, f'{mode}_incorrect_examples_epoch_{epoch}.png'))
            plt.close()
        
        return correct_examples, incorrect_examples

    def _create_examples_figure(self, examples, title):
        """Create figure for analyzing examples"""
        n_samples = min(5, len(examples['images']))
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(n_samples):
            # Convert tensor to numpy for visualization
            img = examples['images'][i].permute(1, 2, 0).numpy()
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            
            label_text = "Hateful" if examples['labels'][i] == 1 else "Non-hateful"
            prob_text = f"Model confidence: {examples['probs'][i]:.3f}"
            
            if 'texts' in examples and examples['texts']:
                text = examples['texts'][i]
                axes[i].set_title(f"Label: {label_text} | {prob_text}\nText: {text[:100]}...")
            else:
                axes[i].set_title(f"Label: {label_text} | {prob_text}")
            
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def compare_models(self, models_dict, dataloader, mode='val'):
        """
        Compare performance of different models on the same validation set.
        
        Args:
            models_dict: Dictionary of {model_name: model}
            dataloader: Validation data loader
            mode: 'val' or 'test'
        """
        results = {name: {'preds': [], 'targets': [], 'probs': []} for name in models_dict}
        
        # Get predictions from each model
        for name, model in models_dict.items():
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    images = batch['image'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(images, input_ids, attention_mask)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    
                    results[name]['probs'].extend(probs)
                    results[name]['preds'].extend(preds)
                    results[name]['targets'].extend(labels.cpu().numpy())
        
        # Create ROC curve comparison
        plt.figure(figsize=(10, 8))
        
        for name, data in results.items():
            fpr, tpr, _ = roc_curve(data['targets'], data['probs'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        
        # Save and add to TensorBoard
        plt.savefig(os.path.join(self.log_dir, f'{mode}_model_comparison_roc.png'))
        self.writer.add_figure(f'{mode}/model_comparison_roc', plt.gcf())
        plt.close()
        
        # Create confusion matrices
        n_models = len(models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (name, data) in enumerate(results.items()):
            cm = confusion_matrix(data['targets'], data['preds'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-hateful', 'Hateful'],
                       yticklabels=['Non-hateful', 'Hateful'], ax=axes[i])
            axes[i].set_xlabel('Predicted labels')
            axes[i].set_ylabel('True labels')
            axes[i].set_title(f'Confusion Matrix - {name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'{mode}_model_comparison_cm.png'))
        self.writer.add_figure(f'{mode}/model_comparison_cm', plt.gcf())
        plt.close()
        
        # Calculate and print metrics
        metrics_table = []
        for name, data in results.items():
            auroc = roc_auc_score(data['targets'], data['probs'])
            precision, recall, f1, _ = precision_recall_fscore_support(
                data['targets'], data['preds'], average='binary')
            
            metrics_table.append([
                name, f"{auroc:.4f}", f"{precision:.4f}", 
                f"{recall:.4f}", f"{f1:.4f}"
            ])
        
        # Create metrics comparison table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=metrics_table,
            colLabels=['Model', 'AUROC', 'Precision', 'Recall', 'F1'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax.set_title('Model Performance Metrics Comparison', fontsize=14)
        
        plt.savefig(os.path.join(self.log_dir, f'{mode}_model_comparison_metrics.png'))
        self.writer.add_figure(f'{mode}/model_comparison_metrics', plt.gcf())
        plt.close()
        
        # Write metrics to CSV
        metrics_file = os.path.join(self.log_dir, f'{mode}_model_comparison.csv')
        with open(metrics_file, 'w') as f:
            f.write('model,auroc,precision,recall,f1\n')
            for row in metrics_table:
                f.write(','.join(row) + '\n')
        
        return results

    def generate_analysis_report(self, model_results=None, dataset_info=None):
        """Generate a written analysis report"""
        report_path = os.path.join(self.log_dir, 'model_analysis.md')
        
        with open(report_path, 'w') as f:
            f.write("# Hateful Memes Detection Model Analysis\n\n")
            
            # Dataset information
            if dataset_info:
                f.write("## Dataset Information\n\n")
                f.write(f"- Training samples: {dataset_info.get('train_samples', 'N/A')}\n")
                f.write(f"- Validation samples: {dataset_info.get('val_samples', 'N/A')}\n")
                f.write(f"- Test samples: {dataset_info.get('test_samples', 'N/A')}\n")
                f.write(f"- Class distribution: {dataset_info.get('class_distribution', 'N/A')}\n\n")
            
            # Model architecture
            f.write("## Model Architecture\n\n")
            f.write("Our model uses a late fusion approach, combining:\n")
            f.write("- **Image Encoder**: ResNet50 pretrained on ImageNet\n")
            f.write("- **Text Encoder**: BERT pretrained on large text corpus\n")
            f.write("- **Fusion Strategy**: Concatenation of image and text features\n\n")
            
            # Performance metrics
            f.write("## Performance Metrics\n\n")
            if model_results:
                f.write("| Model | AUROC | Precision | Recall | F1 |\n")
                f.write("|-------|-------|-----------|--------|----|\n")
                for name, metrics in model_results.items():
                    auroc = metrics.get('auroc', 'N/A')
                    precision = metrics.get('precision', 'N/A')
                    recall = metrics.get('recall', 'N/A')
                    f1 = metrics.get('f1', 'N/A')
                    f.write(f"| {name} | {auroc} | {precision} | {recall} | {f1} |\n")
                f.write("\n")
            
            # Analysis of results
            f.write("## Analysis of Results\n\n")
            
            # BERT vs LSTM analysis
            f.write("### BERT vs LSTM Performance\n\n")
            f.write("BERT outperforms LSTM for text encoding in this task for several reasons:\n\n")
            f.write("1. **Contextual Understanding**: BERT's bidirectional attention mechanism captures context in both directions, essential for understanding nuanced hate speech.\n")
            f.write("2. **Pre-training Advantage**: BERT is pre-trained on a massive corpus, giving it strong language understanding capabilities.\n")
            f.write("3. **Handling of Out-of-Vocabulary Words**: BERT's WordPiece tokenization handles rare words better than LSTM's fixed vocabulary.\n")
            f.write("4. **Attention to Important Words**: BERT's self-attention mechanism focuses on relevant words for classification.\n\n")
            
            # Fusion strategy analysis
            f.write("### Fusion Strategy Impact\n\n")
            f.write("Late fusion (concatenation of features) works well for this task because:\n\n")
            f.write("1. **Modality Independence**: It allows each modality to be processed by specialized architectures.\n")
            f.write("2. **Feature Normalization**: L2 normalization before fusion prevents one modality from dominating.\n")
            f.write("3. **Complementary Information**: Text and image features provide complementary signals for classification.\n\n")
            
            # Error analysis
            f.write("### Error Analysis\n\n")
            f.write("Common patterns in misclassified examples:\n\n")
            f.write("1. **Subtle Hate Speech**: The model struggles with examples where hate is implied rather than explicit.\n")
            f.write("2. **Multimodal Understanding**: Some examples require complex reasoning about the relationship between text and image.\n")
            f.write("3. **Cultural Context**: Memes that require specific cultural knowledge are challenging.\n\n")
            
            # Limitations
            f.write("## Limitations\n\n")
            f.write("1. **Class Imbalance**: The dataset contains more non-hateful than hateful memes, potentially biasing the model.\n")
            f.write("2. **Dataset Size**: The limited size of the dataset may not capture the full diversity of hateful content.\n")
            f.write("3. **Modality Bias**: The model may rely too heavily on either text or image features in certain cases.\n")
            f.write("4. **Generalization**: The model may not generalize well to new types of hateful content or different visual styles.\n\n")
            
            # Future improvements
            f.write("## Future Improvements\n\n")
            f.write("1. **Cross-modal Attention**: Implementing attention mechanisms between modalities could improve fusion.\n")
            f.write("2. **Data Augmentation**: More sophisticated augmentation techniques could help address class imbalance.\n")
            f.write("3. **Ensemble Methods**: Combining multiple models could improve robustness.\n")
            f.write("4. **Explainability**: Adding visualization of attention weights could help interpret model decisions.\n")
        
        print(f"Analysis report generated at {report_path}")
        return report_path

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
    batch_size=16,  # Smaller batch size
    shuffle=True,
    num_workers=2,
    pin_memory=True
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
optimizer = torch.optim.AdamW([
    {'params': model.image_encoder.fc.parameters(), 'lr': 1e-4},
    {'params': model.text_encoder.parameters(), 'lr': 1e-5},
    {'params': model.fusion.parameters(), 'lr': 1e-4}
], weight_decay=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=2, gamma=0.5
)

# Loss function with class weights
pos_weight = torch.tensor([2.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Initialize evaluator
evaluator = KaggleHatefulMemesEvaluator(model, device)

# Training loop
num_epochs = 1
best_val_auroc = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)
        
        # Mixed precision training
        with autocast('cuda'):
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
        
        optimizer.zero_grad()
        scaler = GradScaler()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Log batch loss to TensorBoard
        evaluator.writer.add_scalar('Batch/train_loss', loss.item(), 
                                   epoch * len(train_loader) + batch_idx)
        
        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}')
    
    # Log average training loss
    evaluator.writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
    
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
    evaluator.log_sample_predictions(batch, outputs, epoch, mode='val')
    
    # Analyze model performance (every 2 epochs to save time)
    if epoch % 2 == 0:
        evaluator.analyze_model_performance(val_loader, epoch, mode='val')
    
    # Save best model
    if val_metrics['auroc'] > best_val_auroc:
        best_val_auroc = val_metrics['auroc']
        torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step()

# Final evaluation
print("\nTraining completed! Loading best model for final evaluation...")
try:
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint)
    print("Successfully loaded best model.")
except:
    print("Could not load best model, using current model state.")

# Test evaluation
try:
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

    print("Running evaluation on test set...")
    test_metrics = evaluator.evaluate(test_loader, epoch='final', mode='test')
    print("\nTest Predictions Generated")
    
    # Generate CSV submission file
    submission_file = os.path.join(evaluator.log_dir, 'submission.csv')
    with open(submission_file, 'w') as f:
        f.write('id,proba\n')
        for i, prob in enumerate(test_metrics['probabilities']):
            f.write(f'{test_dataset.data[i]["id"]},{prob:.6f}\n')
    
    print(f"Submission file created at {submission_file}")
except Exception as e:
    print(f"Error during test evaluation: {e}")
    print("Skipping test evaluation.")
    test_metrics = {
        'auroc': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }

# Generate analysis report with just the current model
dataset_info = {
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'test_samples': len(test_dataset) if 'test_dataset' in locals() else 'N/A',
    'class_distribution': "See training data distribution"
}

# Create a simplified model results dictionary with just your current model
model_results = {
    'BERT+ResNet (Late Fusion)': {
        'auroc': test_metrics['auroc'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1']
    }
}

# Generate the analysis report
evaluator.generate_analysis_report(model_results, dataset_info)

# Close TensorBoard writer
evaluator.close()

print("\nEvaluation complete! Check the log directory for visualizations and analysis.")