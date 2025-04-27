
'''
The class for training the digital twin by VAE model
'''
#====================some useful functions==========================

from QPT import QPT
from QPT import *
from numExp_qiskit import NumExp
from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import itertools, random
from qutip import Qobj
from qutip.random_objects import rand_super_bcsz, rand_kraus_map, rand_unitary
from qiskit.quantum_info import random_unitary, Operator, average_gate_fidelity, Kraus, Chi, Choi
from scipy.linalg import sqrtm
import random
from os import listdir
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
from qutip.random_objects import rand_super_bcsz, rand_kraus_map, rand_unitary
from brokenaxes import brokenaxes
from functions import generate_valid_cptp_kraus_operators,get_chiF
from functions import EM_QPT
import torch
import torch.nn.functional as F
import random

class VAE_ML():
    def __init__(self, N,train_loader,test_loader,model,optimizer):  # ,Amatrix,psi_in_idea,obervables

        self.N = N
        self.train_loader= train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        #self.trainnum = train_loader.size()[0]
        
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KL_div
        return loss

    # 训练 VAE
    def train_vae(self, epochs=10, device='cuda',save_model=True):
        self.model.train()
        losslist=[]
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data,) in enumerate(self.train_loader):
                data = data.to(device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar )
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            losslist.append(train_loss)
            if epoch%20==0:
                print(f"Epoch {epoch + 1}, Loss: {train_loss / len(self.train_loader.dataset):.5f}")
            
        if save_model==True:
            torch.save(self.model.state_dict(), './VAE/save_model/'+str(self.N)+'q/model_weights.pth')
        return  losslist
    # 测试 VAE

    def test_vae(self):
        z_space = []
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, in self.test_loader:
                #data = data.to(device)
                recon, mu, logvar = self.model(data)# mu is the latant vector
                #z = model.reparameterize(mu, logvar)
                z_space.append(mu.numpy())
                loss = self.loss_function(recon, data, mu, logvar)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(self.test_loader.dataset):.4f}")
        return z_space
    # 运行 VAE 训练和测试

    def generate_digital_twin(self,latent_dim,sample_num):
        model_name =  './VAE/save_model/'+str(self.N)+'q/model_weights.pth'
        print('loading:', model_name)
        # ==================== the trained model ==========================
        self.model.load_state_dict(torch.load(model_name, weights_only=False))  #
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(sample_num, latent_dim)  # 随机采样 latent vector
            sample = self.model.decode(z).cpu().numpy()
        data_digital = sample
        DT_iden_list = []
        for da in data_digital:
            re_part = da[0]#.reshape(4, 4)
            im_part = da[1]#.reshape(4, 4)
            DT_iden_list.append(re_part + im_part * 1j)
        digital_twin = random.choice(DT_iden_list)
        return  digital_twin






