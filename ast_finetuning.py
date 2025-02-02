import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio.transforms as transforms
from torch.utils.data import Dataset
from torch.nn import functional as F
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

# Ensure the root directory exists
os.makedirs("./SpeechCommands", exist_ok=True)

class SubsetSC(SPEECHCOMMANDS):
    """
    A subset of the SPEECHCOMMANDS dataset for training, validation, and testing.
    """
    def __init__(self, root: str = "./SpeechCommands", subset: str = None):
        super().__init__(root=root, download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [os.path.join(self._path, line.strip()) for line in f]
        
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

def collate_fn(batch):
    """
    Collate function to handle varying lengths of audio by padding waveforms to the max length in the batch.
    """
    waveforms, sample_rates, labels, speaker_ids, utterance_numbers = zip(*batch)
    max_length = max(waveform.shape[1] for waveform in waveforms)
    padded_waveforms = [torch.nn.functional.pad(w, (0, max_length - w.shape[1])) for w in waveforms]
    return torch.stack(padded_waveforms), torch.tensor(sample_rates), labels, speaker_ids, utterance_numbers

# Create dataset objects
train_set = SubsetSC(root="./SpeechCommands", subset="training")
val_set   = SubsetSC(root="./SpeechCommands", subset="validation")
test_set  = SubsetSC(root="./SpeechCommands", subset="testing")

# Create dataloaders with the collate function
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class SpeechCommandsDataset(Dataset):
    """
    Custom dataset class for the Speech Commands dataset.
    """
    def __init__(self, root_dir, subset="training"):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root_dir, download=True)
        self.subset = subset
        self.samples = []
        self.labels = []
        self.label_dict = {}

        # Extract labels from dataset
        for idx in range(len(self.dataset)):
            waveform, sample_rate, label, *_ = self.dataset[idx]
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)  # Create label mapping
            if (subset == "training" and idx % 10 < 8) or (subset == "validation" and idx % 10 == 8) or (subset == "testing" and idx % 10 == 9):
                self.samples.append((waveform, sample_rate, label))
                self.labels.append(self.label_dict[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, sample_rate, label = self.samples[idx]
        label_id = self.labels[idx]

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Compute Mel spectrogram
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128
        )(waveform)

        # Convert to log scale
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # Normalize spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

        # Resize spectrogram to 128x128 for AST input
        spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
        spectrogram = spectrogram.squeeze(0)

        return spectrogram, label_id

# Load Speech Commands Dataset
root_dir = "./SpeechCommands"
train_dataset = SpeechCommandsDataset(root_dir, subset="training")
val_dataset = SpeechCommandsDataset(root_dir, subset="validation")

# Create DataLoaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
model.patch_embed.img_size = (128, 128)
model.default_cfg['img_size'] = 128

# Load pretrained weights for AST
ast_pretrained_weight = torch.load("audioset_16_16_0.4422.pth")
'''Download this from https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1 and place it in the same directory as this script.'''
# Modify the patch embedding layer to match AST
model.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(8, 8), stride=(8, 8))

# Modify the positional embedding to match AST's shape
num_tokens_ast = 257  # 256 patches + 1 CLS token
model.pos_embed = nn.Parameter(torch.randn(1, num_tokens_ast, 768) * .02)

# Update the model's state dict with AST pretrained weights
v = model.state_dict()
v['cls_token'] = ast_pretrained_weight['module.v.cls_token']
# Skip pos_embed since it's already initialized with the right shape
# v['pos_embed'] = ast_pretrained_weight['module.v.pos_embed']  # Do not load this!
# v['patch_embed.proj.weight'] = ast_pretrained_weight['module.v.patch_embed.proj.weight']
# v['patch_embed.proj.bias'] = ast_pretrained_weight['module.v.patch_embed.proj.bias']

for i in range(12):
    v['blocks.' + str(i) + '.norm1.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.norm1.weight']
    v['blocks.' + str(i) + '.norm1.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.norm1.bias']
    v['blocks.' + str(i) + '.attn.qkv.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.attn.qkv.weight']
    v['blocks.' + str(i) + '.attn.qkv.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.attn.qkv.bias']
    v['blocks.' + str(i) + '.attn.proj.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.attn.proj.weight']
    v['blocks.' + str(i) + '.attn.proj.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.attn.proj.bias']
    v['blocks.' + str(i) + '.norm2.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.norm2.weight']
    v['blocks.' + str(i) + '.norm2.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.norm2.bias']
    v['blocks.' + str(i) + '.mlp.fc1.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.mlp.fc1.weight']
    v['blocks.' + str(i) + '.mlp.fc1.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.mlp.fc1.bias']
    v['blocks.' + str(i) + '.mlp.fc2.weight'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.mlp.fc2.weight']
    v['blocks.' + str(i) + '.mlp.fc2.bias'] = ast_pretrained_weight['module.v.blocks.' + str(i) + '.mlp.fc2.bias']
v['norm.weight'] = ast_pretrained_weight['module.v.norm.weight']
v['norm.bias'] = ast_pretrained_weight['module.v.norm.bias']
model.load_state_dict(v)

# Update the Classifier Head
num_speech_commands = 35
in_features = model.head.in_features
model.head = nn.Linear(in_features, num_speech_commands)

# Set Up Device, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Early Stopping Parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Initialize CSV Logging
csv_filename = "training_metrics.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

# Training Loop
num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # Training loop with tqdm
    for spectrograms, labels in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False):
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * spectrograms.size(0)
    
    train_loss = running_loss / len(train_dataset)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation loop with tqdm
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for spectrograms, labels in tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_running_loss += loss.item() * spectrograms.size(0)
    
    val_loss = val_running_loss / len(val_dataset)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Log to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} epochs.")
            break

print("Training complete!")

# Plot Training Metrics
def plot_metrics():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

plot_metrics()