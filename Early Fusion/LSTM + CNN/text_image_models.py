import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

class EarlyFusionLSTMCNNModel(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=300, hidden_dim=256, num_layers=2, 
                 dropout=0.3, bidirectional=True, fusion_dim=512):
        super().__init__()
        
        # Text Encoder: LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.text_dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.text_output_dim = hidden_dim * self.num_directions
        
        # Image Encoder: CNN
        self.image_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 112x112
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 56x56
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        
        # Early fusion: combine features before final processing
        # Calculate the output size of the CNN
        self.cnn_output_size = 128 * 28 * 28
        
        # Cross-modal attention mechanism
        self.use_attention = True
        if self.use_attention:
            self.text_attention = nn.Linear(self.text_output_dim, fusion_dim)
            self.image_attention = nn.Linear(self.cnn_output_size, fusion_dim)
            self.attention_weights = nn.Linear(fusion_dim, 2)
        
        # Fusion layers
        combined_dim = self.text_output_dim + self.cnn_output_size
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
                    
    def forward(self, images, input_ids, attention_mask):
        # Process text with LSTM
        # Calculate sequence lengths from attention mask
        seq_lengths = attention_mask.sum(dim=1).cpu()
        
        # Embed tokens
        embedded = self.embedding(input_ids)
        embedded = self.text_dropout(embedded)
        
        # Pack sequences for LSTM
        packed = pack_padded_sequence(
            embedded, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with LSTM
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(packed)
        
        # Concatenate bidirectional outputs
        if self.num_directions == 2:
            text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            text_features = hidden[-1]
        
        # Process images with CNN
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        
        # Apply cross-modal attention if enabled
        if self.use_attention:
            text_proj = self.text_attention(text_features)
            image_proj = self.image_attention(image_features)
            
            # Calculate attention scores
            fusion_repr = text_proj + image_proj
            attention_scores = F.softmax(self.attention_weights(fusion_repr), dim=1)
            
            # Apply attention weights
            text_features = text_features * attention_scores[:, 0].unsqueeze(1)
            image_features = image_features * attention_scores[:, 1].unsqueeze(1)
        
        # Early fusion: concatenate features
        combined = torch.cat([text_features, image_features], dim=1)
        
        # Classification
        return self.fusion(combined)
