import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score
import math



# Constants
NUM_EPOCHS = 3
BATCH_SIZE = 4  # Keep this small
MAX_SEQ_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 8  # New: This will simulate a batch size of 16
LEARNING_RATE = 5e-5  # Reduced from 5e-4
NUM_WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
NUM_MIND_STATES = 25  # Number of clusters for K-means

# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#cell 2
#%load_ext tensorboard


class MindStateTracker:
    def __init__(self, log_dir='runs/mindstate_experiment'):
        self.writer = SummaryWriter(log_dir)
        self.mind_states = []
        self.mind_state_labels = []
        
    def log_step(self, step, lm_loss, mind_state_loss, total_loss, mind_states, clusters):
        self.writer.add_scalar('Loss/language_model', lm_loss, step)
        self.writer.add_scalar('Loss/mind_state', mind_state_loss, step)
        self.writer.add_scalar('Loss/total', total_loss, step)
        
        # Store mind states for visualization
        # Ensure mind_states is 2D before storing
        if mind_states.dim() == 3:
            mind_states = mind_states.reshape(-1, mind_states.shape[-1])
        self.mind_states.extend(mind_states.detach().cpu().numpy())
        
        # Ensure clusters is 1D
        if clusters.dim() > 1:
            clusters = clusters.reshape(-1)
        self.mind_state_labels.extend(clusters.cpu().numpy())
        
        self.mind_states = []
        self.mind_state_labels = []

class CustomGPT2Model(torch.nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()  # Initialize nn.Module
        # Load base model
        self.model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        
        # Add mind state tokens to vocabulary
        self.mind_state_tokens = [f"<MIND_STATE_{i}>" for i in range(NUM_MIND_STATES)]
        self.tokenizer.add_tokens(self.mind_state_tokens)
        
        # Resize model embeddings to account for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store the indices of mind state tokens
        self.mind_state_token_ids = [
            self.tokenizer.convert_tokens_to_ids(token) 
            for token in self.mind_state_tokens
        ]
        
        # Initialize mind state embeddings randomly
        self.model.transformer.wte.weight.data[self.mind_state_token_ids] = \
            torch.randn(NUM_MIND_STATES, self.model.config.n_embd) * 0.02

    def forward(self, *args, **kwargs):
        # Delegate forward pass to the GPT2 model
        return self.model(*args, **kwargs)

    def get_mind_state_embedding(self, mind_state_idx):
        # Get embedding for a specific mind state
        token_id = self.mind_state_token_ids[mind_state_idx]
        return self.model.transformer.wte.weight[token_id]

class MindStateNetwork(torch.nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        
        # Smaller dimension: 32 instead of 256
        self.projection_dim = 32
        
        # Project from 768+1 to 32 dimensions
        self.hidden_projection = torch.nn.Linear(hidden_size + 1, self.projection_dim)
        
        # Simple attention components
        self.query = torch.nn.Linear(self.projection_dim, self.projection_dim)
        self.key = torch.nn.Linear(self.projection_dim, self.projection_dim)
        self.value = torch.nn.Linear(self.projection_dim, self.projection_dim)
        
        # Final prediction
        self.to_mindstate = torch.nn.Linear(self.projection_dim, NUM_MIND_STATES)
        
        self.scale = math.sqrt(self.projection_dim)

    def forward(self, hidden_states, cluster_ids):
        """
        Predict next mind state using attention over previous states
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            cluster_ids: [batch_size, seq_len]
        Returns:
            mind_state_logits: [batch_size, seq_len, NUM_MIND_STATES]
        """
        # Combine hidden states with their cluster assignments
        cluster_ids = cluster_ids.unsqueeze(-1).float()
        combined = torch.cat([hidden_states, cluster_ids], dim=-1)
        
        # Project to smaller dimension
        projected = self.hidden_projection(combined)  # [batch_size, seq_len, 32]
        
        # Compute Q, K, V
        q = self.query(projected)
        k = self.key(projected)
        v = self.value(projected)
        
        # Compute attention scores and apply scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention, v)
        
        # Predict mind states
        mind_state_logits = self.to_mindstate(attended)
        
        return mind_state_logits

# Load and preprocess dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = CustomGPT2Model("gpt2").to(device) 
tokenizer = model.tokenizer  
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    # Remove samples that are mostly <|endoftext|>
    valid_samples = (tokenized['input_ids'] != tokenizer.eos_token_id).sum(dim=1) > MAX_SEQ_LENGTH // 2
    return {k: v[valid_samples] for k, v in tokenized.items()}

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create DataLoader
data_loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)



def fit_kmeans(model, data_loader, num_samples=2500):
    """
    Fits KMeans on hidden states from the model to create mind state clusters.
    Collects num_samples embeddings and clusters them into NUM_MIND_STATES groups.
    """
    model.eval()
    hidden_states_collection = []
    samples_collected = 0
    
    print("Collecting hidden states for KMeans...")
    with torch.no_grad():
        for batch in data_loader:
            # Get model outputs
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.model(inputs, attention_mask=attention_mask, output_hidden_states=True)
            
            # Get hidden states and add them to our collection
            hidden_states = outputs.hidden_states[-3]  # Shape: [batch_size, seq_len, hidden_dim]
            hidden_states = hidden_states.cpu().numpy().reshape(-1, hidden_states.shape[-1])
            
            # Add to collection
            hidden_states_collection.append(hidden_states)
            samples_collected += hidden_states.shape[0]
            
            if samples_collected >= num_samples:
                break
    
    # Combine all states
    all_states = np.vstack(hidden_states_collection)
    
    # Take only num_samples states if we collected more
    if len(all_states) > num_samples:
        indices = np.random.choice(len(all_states), num_samples, replace=False)
        all_states = all_states[indices]
    
    print(f"Fitting KMeans on {len(all_states)} states into {NUM_MIND_STATES} clusters...")
    kmeans = KMeans(n_clusters=NUM_MIND_STATES, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(all_states)
    
    # Log the distribution of samples across clusters
    cluster_labels = kmeans.labels_
    cluster_distribution = np.bincount(cluster_labels, minlength=NUM_MIND_STATES)
    print(f"Cluster Distribution after KMeans fitting: {cluster_distribution.tolist()}")
    
    return kmeans


# After model creation
mind_state_net = MindStateNetwork().to(device)
optimizer = AdamW([
    {'params': model.parameters()},
    {'params': mind_state_net.parameters()}
], lr=LEARNING_RATE)
total_steps = len(data_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=total_steps)


def train_step(batch):
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Get hidden states from GPT2
    outputs = model.model(inputs, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-3]
    
    # Get cluster assignments from KMeans
    with torch.no_grad():
        hidden_np = hidden_states.cpu().numpy().reshape(-1, hidden_states.shape[-1])
        cluster_ids = kmeans.predict(hidden_np)
        # Convert to Long tensor explicitly
        cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=device).view(hidden_states.shape[0], -1)
    
    # Predict next mind states
    mind_state_logits = mind_state_net(hidden_states, cluster_ids)
    
    # Get target clusters for mind state loss
    target_clusters = cluster_ids[:, 1:]  # remove first position
    predicted_logits = mind_state_logits[:, :-1, :]  # remove last position
    
    # Calculate mind state loss - ensure targets are Long
    mind_state_loss = F.cross_entropy(
        predicted_logits.reshape(-1, NUM_MIND_STATES),
        target_clusters.reshape(-1).long(),  # explicitly convert to long
        ignore_index=-100
    )
    
    # Get predicted mind state token for the sequence
    next_cluster = torch.argmax(mind_state_logits[:, -1], dim=-1)
    mind_state_tokens = torch.tensor([
        model.mind_state_token_ids[idx] for idx in next_cluster
    ], dtype=torch.long, device=device).unsqueeze(1)
    
    # Extend attention mask and labels for the additional token
    extended_attention_mask = torch.cat([
        attention_mask,
        torch.ones(attention_mask.shape[0], 1, device=device)
    ], dim=1)
    
    extended_labels = torch.cat([
        inputs,
        torch.full((inputs.shape[0], 1), -100, dtype=torch.long, device=device)
    ], dim=1)
    
    # Forward pass with mind state token
    outputs = model.model(
        input_ids=torch.cat([inputs, mind_state_tokens], dim=1),
        attention_mask=extended_attention_mask,
        labels=extended_labels
    )
    
    # Combine losses
    total_loss = outputs.loss + 0.5 * mind_state_loss
    
    return total_loss, outputs.loss, mind_state_loss, next_cluster, mind_state_logits


# Training loop
tracker = MindStateTracker()

# After model creation but before training loop
print("Fitting initial KMeans clusters...")
kmeans = fit_kmeans(model, data_loader)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_lm_loss = 0
    total_mind_state_loss = 0
    optimizer.zero_grad()  # Move outside the batch loop
    
    for step, batch in enumerate(data_loader):
        # Get losses from train_step
        loss, lm_loss, mind_state_loss, quantized_states, next_mind_state = train_step(batch)
        
        # Scale the loss
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        # Log to tensorboard
        global_step = epoch * len(data_loader) + step
        tracker.log_step(
            global_step, 
            lm_loss.item(),
            mind_state_loss.item(),
            loss.item(),
            next_mind_state.detach(),
            quantized_states
        )
        
        # Only optimize every GRADIENT_ACCUMULATION_STEPS
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Retrain K-means every 500 steps
        if step > 0 and step % 500 == 0:
            print(f"\nStep {step}: Retraining KMeans...")
            kmeans = fit_kmeans(model, data_loader)
        
        if step % 25 == 0:
            torch.cuda.empty_cache()
            print(f"Epoch {epoch+1}, Step {step}")
            print(f"  Total Loss: {loss:.4f}")
            print(f"  LM Loss: {lm_loss:.4f}")
            print(f"  Mind State Loss: {mind_state_loss:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Mind State Distribution: {quantized_states.tolist()[:10]}")
            print(f"  Sample Input:", tokenizer.decode(batch["input_ids"][0][:30]).replace(tokenizer.pad_token, ""))
            


# Save the model and mind state embedder
model.save_pretrained("./gpt2_with_mind_states")
tokenizer.save_pretrained("./gpt2_with_mind_states")
