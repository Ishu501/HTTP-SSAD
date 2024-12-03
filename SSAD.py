import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- Load Dataset ---
def load_data(file_path):
    """
    Load the dataset from a .txt file where each HTTP request is a separate line.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

# --- Preprocessing (BERT Tokenization and Embeddings) ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_and_vectorize(requests):
    """
    Tokenize the HTTP requests and convert them into BERT embeddings.
    """
    inputs = tokenizer(requests, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# --- Autoencoder Model ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Training Loop ---
def train_autoencoder(model, data, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the autoencoder model on the dataset.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = torch.stack(batch).float()  # Convert batch to tensor
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(data)}")

# --- Anomaly Detection ---
def detect_anomalies(model, data, threshold=0.1):
    """
    Detect anomalies by checking the reconstruction error.
    """
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)
        reconstruction_error = torch.mean((data - reconstructions) ** 2, dim=1)
    
    # Flag data as anomaly if reconstruction error exceeds threshold
    anomalies = reconstruction_error > threshold
    return anomalies

# --- K-Fold Cross Validation ---
def kfold_cross_validation(X, k=4, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Perform K-fold cross-validation on the dataset.
    """
    kfold = KFold(n_splits=k, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Training fold {fold+1}")
        X_train, X_val = X[train_idx], X[val_idx]
        
        # Train the model on this fold
        train_autoencoder(autoencoder, X_train, epochs, batch_size, learning_rate)
        
        # Optionally evaluate on the validation set
        # You can add evaluation logic here (e.g., compute reconstruction error)
        
# --- FastAPI Integration ---
app = FastAPI()

class RequestData(BaseModel):
    requests: list[str]

@app.post("/detect")
def detect(request_data: RequestData):
    """
    Detect anomalies for new HTTP requests via FastAPI.
    """
    requests = request_data.requests
    embeddings = tokenize_and_vectorize(requests)
    anomalies = detect_anomalies(autoencoder, embeddings)
    return {"anomalies": anomalies.tolist()}

# --- Main Function ---
if __name__ == "__main__":
    # Load the dataset (CSIC-2010 dataset)
    file_path = 'data/datasets/CSIC-2010/normalTrafficTraining.txt'  # Update this with the path to your CSIC-2010 .txt file
    http_requests = load_data(file_path)

    # Convert the HTTP requests into BERT embeddings
    X = tokenize_and_vectorize(http_requests)
    
    # Initialize the Autoencoder model
    input_dim = X.shape[1]  # Dimension of the input (number of features)
    autoencoder = Autoencoder(input_dim)
    
    # Train the model with K-fold cross-validation
    kfold_cross_validation(X, k=4, epochs=10, batch_size=32, learning_rate=0.001)

    # Run the FastAPI app for real-time anomaly detection
    uvicorn.run(app, host="0.0.0.0", port=8000)
