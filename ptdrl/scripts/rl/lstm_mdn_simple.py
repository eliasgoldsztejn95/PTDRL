import os
import torch
import torch.nn as nn
import numpy as np
from early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import math
# Standard Library


# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

torch.manual_seed(0)
ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

# open np file
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_checkpoint = dir_path
dir_path_checkpoint += "/checkpoint.pt"

dir_path_np = dir_path
dir_path_np += "/tracking2.npy"


####################
# LSTM Autoencoder #
####################      
# code inspired by  https://github.com/shobrook/sequitur/blob/master/sequitur/autoencoders/rae.py
# annotation sourced by  ttps://pytorch.org/docs/stable/nn.html#torch.nn.LSTM        

# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.fc1 = nn.Linear(self.hidden_size, output_size)
        self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        mu = self.fc1(x)
        sigma = self.fc2(x)
        sigma = 10 + 1e-3 + 10*self.tanh(sigma)
        return mu, sigma
    
# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(self, seq_len = 3, no_features = 12, embedding_dim = 32, learning_rate= 1e-3, every_epoch_print= 100, epochs_mu= 800,
     epochs_sigma= 800, patience= 200, max_grad_norm= 0.005):
        super().__init__()

        self.seq_len = seq_len
        print(f"seq len {self.seq_len}")
        self.no_features = no_features
        print(f"no features {self.no_features}")
        self.embedding_dim = embedding_dim
        print(f"embedding_dim {self.embedding_dim}")

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features)
        
        self.epochs_mu = epochs_mu
        self.epochs_sigma = epochs_sigma
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.every_epoch_print = every_epoch_print
    
    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        mu, sigma = self.decoder(encoded)
        return encoded, mu, sigma
    
    def criterion_mdn(self, mu, sigma, future):
        """Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.
        """
        #print(f"sigma {sigma}")
        prob = self.gaussian_probability(sigma, mu, future)
        #print(f"prob {prob}")
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)

    def gaussian_probability(self, sigma, mu, future):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`"""
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((future - mu) / sigma)**2) / sigma
        #print(f"here {ret}")
        return torch.mean(ret, 2)


    
    def fit(self, x):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """
        # Each sequence contains seq_len elements. The first seq_len/2 elements are the past, and the last seq_len/2 elements are the future
        past, future = torch.split(x,int(self.seq_len),dim=1)
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        crit = nn.MSELoss(reduction='mean')
        self.train()
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)

        for epoch in range(1 , self.epochs_mu + self.epochs_sigma +1):
            # updating early_stopping's epoch
            early_stopping.epoch = epoch        
            optimizer.zero_grad()
            encoded, mu, sigma = self(past)
            loss = 0
            if epoch < self.epochs_mu:
              loss = crit(mu, future)
            else:
              loss = self.criterion_mdn(mu, sigma , future)
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(loss, self)
            
            #if early_stopping.early_stop:
             #   break
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
            optimizer.step()
            
            if epoch % self.every_epoch_print == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")
        
        # load the last checkpoint with the best model
        #self.load_state_dict(torch.load('./checkpoint.pt'))
        
        # to check the final_loss
        encoded, mu, sigma = self(past)
        final_loss = self.criterion_mdn(mu, sigma , future).item()
        
        return final_loss
    
    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        self.eval()
        mu, sigma = self.decoder(x)
        return mu, sigma
    
    def loss(self, x):
        past, future = torch.split(x,int(self.seq_len),dim=1)
        encoded = self.encode(past)
        mu, sigma = self.decode(encoded)

        crit = nn.MSELoss(reduction='mean')
        loss_mse = crit(mu, future)
        loss_mdn = self.criterion_mdn(mu, sigma, future)

        return loss_mse, loss_mdn


    
    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))


class LSTM_MDN():
    def __init__(self, checkpoint):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.checkpoint = checkpoint
        self.model = LSTM_AE()
        print("Loading network")
        self.model.load(self.checkpoint)

    def prepare_dataset(self, sequential_data) :
        if type(sequential_data) == np.array:
            print('array')
            data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
            unsqueezed_data = data_in_tensor.unsqueeze(2)
        elif type(sequential_data) == list:
            print('list')
            data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
            unsqueezed_data = data_in_tensor.unsqueeze(2)
        elif type(sequential_data) == np.ndarray:
            print('ndarray')
            data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
            unsqueezed_data = data_in_tensor
            
        seq_len = unsqueezed_data.shape[1]
        no_features = unsqueezed_data.shape[2] 
        # shape[0] is the number of batches
        
        return unsqueezed_data, seq_len, no_features

    def predict(self, obs):

        refined_input_data, seq_len, no_features = self.prepare_dataset(obs)
        
        embedded_points = self.model.encode(refined_input_data)
        decoded_points_mu, decoded_points_sigma = self.model.decode(embedded_points)

        return decoded_points_mu, decoded_points_sigma
    
###############
# GPU Setting #
###############
#os.environ["CUDA_VISIBLE_DEVICES"]="0"   # comment this line if you want to use all of your GPUs



####################
# Data preparation #
####################
def prepare_dataset(sequential_data) :
    if type(sequential_data) == np.array:
        print('array')
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == list:
        print('list')
        data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == np.ndarray:
        print('ndarray')
        data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
        unsqueezed_data = data_in_tensor
        
    seq_len = unsqueezed_data.shape[1]
    no_features = unsqueezed_data.shape[2] 
    # shape[0] is the number of batches
    
    return unsqueezed_data, seq_len, no_features


##################################################
# QuickEncode : Encoding & Decoding & Final_loss #
##################################################
def QuickEncode(input_data, 
                embedding_dim, 
                learning_rate = 1e-3, 
                every_epoch_print = 100, 
                epochs_mu = 800, 
                epochs_sigma = 800,
                patience = 200, 
                max_grad_norm = 0.005):
    
    refined_input_data, seq_len, no_features = prepare_dataset(input_data)
    model = LSTM_AE(int(seq_len/2), no_features, embedding_dim, learning_rate, every_epoch_print, epochs_mu, epochs_sigma, patience, max_grad_norm)
    final_loss = model.fit(refined_input_data)

    # recording_results
    past, future = torch.split(refined_input_data,int(seq_len/2),dim=1)
    embedded_points = model.encode(past)
    decoded_points_mu, decoded_points_sigma = model.decode(embedded_points)

    return embedded_points.cpu().data, decoded_points_mu.cpu().data,decoded_points_sigma.cpu().data,  model

def example():

    x = np.linspace(0,4*np.pi,20)
    y = np.zeros([500,20,1])
    for i in range(500):
        y[i,:,0] = np.sin(x) + np.random.normal(0,0.1,20)

    encoded, decoded_mu, decoded_sigma, final_loss  = QuickEncode(y, embedding_dim=2)
    print(encoded)
    print(decoded_mu)
    print(decoded_sigma)
    plt.plot(x[9:-1],decoded_mu[8,:,:])
    plt.show()

def main():

    with open(dir_path_np, 'rb') as f:
        y = np.load(f)
    
    shape = y.shape
    
    y_train = y[:int(shape[0]*0.8),:,:]
    y_test = y[int(shape[0]*0.8)+1:-1,:,:]

    print(y_train.shape)
    print(y_test.shape)

    encoded, decoded_mu, decoded_sigma, model  = QuickEncode(y_train, embedding_dim=32)

    refined_input_data, seq_len, no_features = prepare_dataset(y_test)
    mse, mdn = model.loss(refined_input_data)
    print(mse)
    print(mdn)

def main_2():

    a = LSTM_MDN(dir_path_checkpoint)

    with open(dir_path_np, 'rb') as f:
        y = np.load(f)
    
    shape = y.shape
    
    y_train = y[:int(shape[0]*0.8),:,:]
    y_test = y[int(shape[0]*0.8)+1:-1,:,:]
    print(y_test.shape)

    #print(a.model.decode(y))
    refined_input_data, seq_len, no_features = prepare_dataset(y_test)
    print(refined_input_data.shape)
    print(a.model.loss(refined_input_data))

    past, future = torch.split(refined_input_data,int(3),dim=1)
    print(past.shape)
    print(y_test[0:1,0:3,:])
    print(a.predict(y_test[0:1,0:3,:]))

if __name__ == '__main__':
    main_2()