import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import torchvision.models as models

class LSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.2, bidirectional=True):
        """
        LSTM-based text encoder
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMTextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            lengths: Sequence lengths for packing (optional)
            
        Returns:
            output: Last hidden state [batch_size, hidden_dim*num_directions]
        """
        # Embed tokens
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            self.lstm.flatten_parameters()
            _, (hidden, _) = self.lstm(packed)
        else:
            # Otherwise, use standard LSTM
            self.lstm.flatten_parameters()
            output, (hidden, _) = self.lstm(embedded)
        
        # Concatenate bidirectional outputs
        if self.num_directions == 2:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
            
        return hidden

class BERTTextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', freeze_bert=False):
        """
        BERT-based text encoder
        
        Args:
            pretrained_model: Name of the pretrained BERT model
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTTextEncoder, self).__init__()
        
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.output_dim = self.bert.config.hidden_size
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            cls_output: CLS token embedding [batch_size, hidden_size]
        """
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token as the sentence representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return cls_output

class CNNImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        """
        CNN-based image encoder
        
        Args:
            out_dim: Output dimension
        """
        super(CNNImageEncoder, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, out_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        self.output_dim = out_dim
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            features: Image features [batch_size, out_dim]
        """
        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch_size, 128, 28, 28]
        
        # Flatten
        x = x.view(x.size(0), -1)  # -> [batch_size, 128*28*28]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)
        
        return features

class ResNetImageEncoder(nn.Module):
    def __init__(self, pretrained=True, out_dim=512):
        """
        ResNet-based image encoder
        
        Args:
            pretrained: Whether to use pretrained weights
            out_dim: Output dimension
        """
        super(ResNetImageEncoder, self).__init__()
        
        # Use newer ResNet initialization
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final layers and add custom head
        modules = list(resnet.children())[:-2]  # Remove avg pool and fc
        self.features = nn.Sequential(*modules)
        
        # Add custom pooling and FC layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Doubled feature size due to concat of avg and max pool
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim)
        )
        
        self.output_dim = out_dim
        
    def forward(self, x):
        x = self.features(x)
        
        # Combine average and max pooling
        avg_pooled = self.avg_pool(x).flatten(1)
        max_pooled = self.max_pool(x).flatten(1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        features = self.fc(pooled)
        return F.normalize(features, p=2, dim=1)  # L2 normalize features
