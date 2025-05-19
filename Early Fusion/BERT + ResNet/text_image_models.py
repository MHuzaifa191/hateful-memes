import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

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

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism to weigh text/image contributions"""
    def __init__(self, feature_dim):
        super().__init__()
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        self.value_transform = nn.Linear(feature_dim, feature_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        
    def forward(self, text_features, image_features):
        # Transform features
        queries = self.query_transform(text_features)
        keys = self.key_transform(image_features)
        values = self.value_transform(image_features)
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)
        
        return attended_values

class EarlyFusionHatefulMemesModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', use_cross_modal_attention=True):
        super().__init__()
        
        # 1. Image encoder - ResNet50 with intermediate features
        self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully connected layer to get penultimate features
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-2])
        self.image_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.image_projection = nn.Linear(2048, 768)  # Project to match BERT dimension
        
        # 2. Text encoder - BERT with intermediate features
        self.text_encoder = BertModel.from_pretrained(bert_model)
        
        # 3. Fine-tuning configuration
        # Unfreeze the last few layers of ResNet
        for name, param in self.image_encoder.named_parameters():
            if 'layer4' in name:  # Only fine-tune the last layer
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # Unfreeze the last few layers of BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # Only unfreeze the last 2 layers
        for param in self.text_encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # 4. Cross-modal attention (optional)
        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention:
            self.text_to_image_attention = CrossModalAttention(768)
            self.image_to_text_attention = CrossModalAttention(768)
            # Fusion dimension is doubled due to bidirectional attention
            fusion_dim = 768 * 4  # text + image + text2img + img2text
        else:
            fusion_dim = 768 * 2  # text + image
        
        # 5. Early fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        # 6. Better initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images, input_ids, attention_mask):
        # 1. Extract image features
        image_features = self.image_encoder(images)  # [batch_size, 2048, h, w]
        image_features = self.image_pooling(image_features).squeeze(-1).squeeze(-1)  # [batch_size, 2048]
        image_features = self.image_projection(image_features)  # [batch_size, 768]
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 2. Extract text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get CLS token embedding and all hidden states
        cls_embedding = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        text_features = F.normalize(cls_embedding, p=2, dim=1)
        
        # 3. Apply cross-modal attention if enabled
        if self.use_cross_modal_attention:
            # Reshape for attention
            batch_size = text_features.size(0)
            text_features_2d = text_features.view(batch_size, 1, -1)  # [batch_size, 1, 768]
            image_features_2d = image_features.view(batch_size, 1, -1)  # [batch_size, 1, 768]
            
            # Apply bidirectional attention
            text_attended_image = self.text_to_image_attention(text_features_2d, image_features_2d).squeeze(1)
            image_attended_text = self.image_to_text_attention(image_features_2d, text_features_2d).squeeze(1)
            
            # Combine all features
            combined = torch.cat([
                text_features, 
                image_features, 
                text_attended_image, 
                image_attended_text
            ], dim=1)
        else:
            # Simple concatenation
            combined = torch.cat([text_features, image_features], dim=1)
        
        # 4. Pass through fusion classifier
        return self.fusion_classifier(combined)
