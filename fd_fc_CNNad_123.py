import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datetime import datetime, timedelta
from tqdm import tqdm
import xarray as xr
from datetime import datetime
import random
print("Running with torch version:", torch.__version__)

# SEED FIXA PARA REPRODUTIBILIDADE
SEED = 42 

# Python
random.seed(SEED)

# NumPy
#np.random.seed(SEED)

# PyTorch
#torch.manual_seed(SEED)                      # Fixa o gerador de números aleatórios do PyTorch (CPU)
#torch.cuda.manual_seed(SEED)                 # Fixa para CUDA (se estiver usando GPU)
#torch.backends.cudnn.deterministic = True    # Força algoritmos determinísticos
#torch.backends.cudnn.benchmark = False       # Desativa auto-tuning que pode variar entre execuções


## 
# DataSet processing 
##

import numpy as np
import xarray as xr
from PIL import Image
import cv2
from skimage.transform import resize

# === Diretório das imagens TIF (máscaras RGB das frentes) ===
image_dir = "../../shared/tiff_joinyears/"  # <-- 

# === Função para processar máscara RGB ===
def preprocess_mask(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((320, 320), resample=Image.NEAREST)
    img_np = np.array(image) #/ 255 # Descomentar caso decido normalizar.

    """
    mask = np.zeros((320, 320), dtype=np.uint8)
    BLUE = (0, 0, 238)
    RED = (255, 0, 0)
    PURPLE = (145, 44, 238)

    mask[np.all(img_np == BLUE, axis=-1)] = 1
    mask[np.all(img_np == RED, axis=-1)] = 2
    mask[np.all(img_np == PURPLE, axis=-1)] = 3
    
    """
    class_map = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    # Define os intervalos de cor para cada tipo de frente
    BLUE_MIN   = np.array([0, 0, 10])
    BLUE_MAX   = np.array([10, 10, 255])

    RED_MIN    = np.array([10, 0, 0])
    RED_MAX    = np.array([255, 10, 10])

    PURPLE_MIN = np.array([35, 35, 225])
    PURPLE_MAX = np.array([156, 49, 256])

    # Máscaras booleanas por cor
    is_blue   = np.all((img_np >= BLUE_MIN) & (img_np <= BLUE_MAX), axis=-1)
    is_red    = np.all((img_np >= RED_MIN) & (img_np <= RED_MAX), axis=-1)
    is_purple = np.all((img_np >= PURPLE_MIN) & (img_np <= PURPLE_MAX), axis=-1)

    class_map[is_blue] = 1    # Frente fria
    class_map[is_red] = 2     # Frente quente
    class_map[is_purple] = 3  # Frente ocluída 

    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(class_map, kernel, iterations=1)
    return mask_dilated

# === Carregar todas as máscaras ===
def load_all_masks():
    paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")])
    return np.stack([preprocess_mask(p) for p in paths])

# === Função para interpolar e normalizar variáveis do ERA5 ===
def process_variable(var):
    normed = []
    for i in range(var.shape[0]):
        data = var[i].values
        data_resized = resize(data, (320, 320), order=1, mode='edge', anti_aliasing=False)
        data_norm = (data_resized - np.min(data_resized)) / (np.max(data_resized) - np.min(data_resized) + 1e-6)
        normed.append(data_norm)
    return np.stack(normed)

# === Função principal para carregar dados ===

def load_data(netcdf_path):
    ds = netcdf_path  #xr.open_dataset(netcdf_path)

    msl = process_variable(ds['msl'])
    t2m = process_variable(ds['t2m'])
    d2m = process_variable(ds['d2m'])
    u10 = process_variable(ds['u10'])
    v10 = process_variable(ds['v10'])
    wind_mag = np.sqrt(u10**2 + v10**2)

    X = np.stack([msl, t2m, wind_mag, d2m], axis=1)  # Shape: (8, 4, 320, 320)
    Y = load_all_masks()  # Shape: (8, 320, 320)
    return X, Y

# === X and Y definition ===
# image_dir = "./imagens_tif"
# X, masks = load_data("netcdf_file.nc")

#ncfile = "../../shared/data_stream-oper_stepType-instant.nc"
#load_all_masks() ## Read masks
#X, masks = load_data(ncfile)

# Era5 files concatenations 
f18 = "../../shared/2018.nc"  
f19 = "../../shared/2019_singlelevel_6dca10e42f24ed55f8a735855e66c52c.nc"
f20 = "../../shared/2020_singleLevel_data_stream-oper_stepType-instant.nc"

file_paths = [f18, f19, f20]
combined_dataset = xr.open_mfdataset(file_paths)
combined_dataset = xr.open_mfdataset(file_paths, combine='nested', concat_dim='valid_time')

ncfile = combined_dataset # "../../shared/data_stream-oper_stepType-instant.nc"
load_all_masks() ### Read masks
X, masks = load_data(ncfile)


##
# Model Simple Unet
##
# ==== Dataset Personalizado ====
class FrontDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==== CNN Profissional ====
class AdvancedCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(AdvancedCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==== Função de Treinamento ====
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Cálculo da Acurácia de Treinamento (Pixel-wise)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == Y_batch).sum().item()
        total_pixels += Y_batch.numel()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_pixels
    return avg_loss, avg_accuracy

# ==== Função de Validação ====
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
            
            # Cálculo da Acurácia de Validação (Pixel-wise)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == Y_batch).sum().item()
            total_pixels += Y_batch.numel()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_pixels
    return avg_loss, avg_accuracy

# ==== Função Principal ====
def main_cnnad(X, Y):
    # Separar treino/teste
    X_train, X_val = X[:4100], X[4100:]
    Y_train, Y_val = Y[:4100], Y[4100:]

    train_dataset = FrontDataset(X_train, Y_train)
    val_dataset = FrontDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    
    # NOVAS VARIÁVEIS PARA ACURÁCIA
    train_accuracies = []
    val_accuracies = []
    
    # datetime definition
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d_%H%M") # Example: 20250807
    
    # --- Lines for Best Model and Early Stop / LINHAS PARA EARLY STOPPING E MELHOR MODELO ---
    best_val_loss = float('inf')  # Inicializa a melhor perda de validação com infinito
    patience = 99                 # Define o número de épocas para esperar
    patience_counter = 0          # Contador de épocas sem melhoria
    #best_model_path = ""          # Variável para armazenar o caminho do melhor modelo
    # --------------------------------------------------------
    best_model_path = f"fd_fc_CNNad_123/cnn_ad_best_model_{date_string}.pth" # Melhor Modelo
    print(f"Best Model Saved *_{date_string}")
    
    for epoch in range(3):
        # Treinamento
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        # Validação
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Imprime as novas métricas
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Armazena as métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # --- SALVAR O MELHOR MODELO E EARLY STOPPING ---
        
        '''
        O melhor modelo é salvo sempre que a perda de validação melhora. Mas o loop de treinamento
        não é interrompido porque eu quero que ele rode todas as épocas,
        '''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reseta o contador
            torch.save(model.state_dict(), best_model_path) # Salva o novo melhor modelo
            print(f"--> Val Loss melhorou. Modelo salvo em: {best_model_path}")
        else:
            patience_counter += 1 # Incrementa o contador
            print(f"--> Val Loss não melhorou. Paciência: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early Stopping! Nenhuma melhoria em {patience} épocas. Parando o treinamento.")
            #break # Commentado para não interromper o loop de treinamento (Disabilitando o Early Stopping)
        # --------------------------------------------------------

    
    torch.save(model.state_dict(), f"fd_fc_CNNad_123/cnn_ad_model_{date_string}.pth")
    print("Modelo salvo como cnn_model.pth")
    
    # Carregar o MELHOR modelo para as predições e visualização.
    print(f"\nCarregando o melhor modelo salvo: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path)) # Carrega os pesos do melhor modelo
    

    # Predição de exemplo
    model.eval()
    sample_X = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(sample_X).argmax(1).cpu().numpy()
        
    
     # ==== Visualização das métricas ====
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1) # Mudança para 1 linha, 2 colunas, posição 1
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2) # Posição 2
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"fd_fc_CNNad_123/cnnad123_metric_plot_{date_string}.png")
    

    ## Plot of the images to compare GroundTruth and Predictions
    for i in range(len(pred)):
        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1)
        plt.title(f"X[{i}] Values")
        plt.imshow(X_val[i][3], cmap="viridis")
        plt.subplot(1,3,2)
        plt.title(f"GroundTruth Y[{i}]")
        plt.imshow(Y_val[i], cmap="viridis")
        plt.subplot(1,3,3)
        plt.title(f"PrediçãoCNN pr[{i}]")
        plt.imshow(pred[i], cmap="viridis")
        plt.tight_layout()
        plt.savefig(f"fd_fc_CNNad_123/unet_{i}_{date_string}.png")

# === Execução ===

main_cnnad(X, masks)
