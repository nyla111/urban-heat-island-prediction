import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ..config import LSTM_PARAMS



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out



def prepare_lstm_data(uhi_data, look_back=20):
    """Prepare data for LSTM"""
    features_to_keep = ['coastal', 'red', 'green', 'blue', 'nir08', 'swir16', 'NDVI']
    X_raw = uhi_data[features_to_keep].values
    y_raw = uhi_data['UHI'].values
    
    # Normalize
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1))
    

    # Create sequences
    def create_sequences(X, y, look_back):
        X_seq, y_seq = [], []
        for i in range(len(X) - look_back):
            X_seq.append(X[i:i+look_back])
            y_seq.append(y[i+look_back])
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, look_back)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_scaler, y_scaler


def train_lstm(model, X_train, y_train, num_epochs=2000, learning_rate=0.005):
    """Train LSTM model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return loss_list