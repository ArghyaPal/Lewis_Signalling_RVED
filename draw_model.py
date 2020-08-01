import torch
import torch.nn as nn
from utility import *
import torch.functional as F
import torch.optim as optim

import time

import torch
from matplotlib import pyplot as plt

class DrawModel(nn.Module):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size, train_loader):
        super(DrawModel,self).__init__()
        self.T = T
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.cs_d_ = [0] * T
        self.cs_rec_ = [0] * T
        self.cs_decoder = [0] * T


        self.logsigmas,self.sigmas,self.mus = [0] * T,[0] * T,[0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size, dec_size) #nn.LSTMCell(2 * 5 + dec_size, dec_size)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.enc_w_linear = nn.Linear(dec_size, N*N)
        self.generator = nn.Sequential(
                                nn.Linear(A*B, A*B),
                                nn.ReLU())
        self.generator = self.generator.cuda()

        self.train_loader = train_loader
        self.sigmoid = nn.Sigmoid()


    def normalSample(self):
        return Variable(torch.randn(self.batch_size,self.z_size))

    # correct
    def compute_mu(self,g,rng,delta):
        rng_t,delta_t = align(rng,delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t,g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    # correct
    def filterbank(self,gx,gy,sigma2,delta):
        rng = Variable(torch.arange(0,self.N).view(1,-1))
        mu_x = self.compute_mu(gx,rng,delta)
        mu_y = self.compute_mu(gy,rng,delta)

        a = Variable(torch.arange(0,self.A).view(1,1,-1))
        b = Variable(torch.arange(0,self.B).view(1,1,-1))

        mu_x = mu_x.view(-1,self.N,1)
        mu_y = mu_y.view(-1,self.N,1)
        sigma2 = sigma2.view(-1,1,1)

        Fx = self.filterbank_matrices(a,mu_x,sigma2)
        Fy = self.filterbank_matrices(b,mu_y,sigma2)

        return Fx,Fy


    def forward(self,x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))
        ram = Variable(torch.zeros(self.batch_size, self.enc_size), requires_grad=False)

        h_enc = Variable(torch.zeros(self.batch_size, self.enc_size))
        h_dec = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))

        loss = 0
        Lx = 0
        La = 0
        Lb = 0
        criterion = nn.MSELoss()
        criterion_1 = nn.L1Loss()
        
        for t in range(self.T):
            ###################################################
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            c_prev_decoder = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs_decoder[t-1]

            x_hat = x - self.sigmoid(c_prev)
            r_t = self.read(x,x_hat,h_enc_prev)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t,h_enc_prev),1), (h_enc,enc_state))
            self.cs[t] = c_prev + self.write(h_enc_prev)
            ram = c_prev
            ram = ram.detach()
            ###################################################
            
            s_t, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc_prev)
            ###################################################


            h_dec, dec_state = self.decoder(s_t, (h_dec_prev, dec_state)) 
            x_guess = self.generator(self.write(h_dec))
            ###################################################
            # We are giving a lightweight KNN
            # x_pseudo =  self.KNN(x_guess)  
            # We used the KNN based retrieval in our final implementation, 
            #Howvever, in this implementation we are using ground truth itself
            x_pseudo = x

            if t % 3 == 0:
              Lx += criterion(x_guess, x_pseudo)
              #La += criterion(self.write(h_dec), ram)
              #print(Lx)
              Lb += criterion(x_guess, ram)
              #print(Lb)
            
            self.cs_rec_[t] =  x_guess # c_prev_decoder is working fine ...!
            h_dec_prev = h_dec

        La = criterion_1(x_guess, ram)
        imgs_gen = []
        for img_gen in self.cs_rec_:
            imgs_gen.append(self.sigmoid(img_gen).cpu().data.numpy())
        
        return Lx, La, Lb, imgs_gen, self.cs_rec_ #x_guess

# This is not available
    def KNN(self, x_guess):   # We will give the full version in our official release after acceptance
         dist = F.cosine_similarity(self.train_loader, x_guess)
         index_sorted = torch.argsort(dist)
         top_1 = index_sorted[:1]
         return x_pseudo


    def loss(self, x):
        Lx, La, Lb, imgs_rec, self.cs_rec_ = self.forward(x)

        criterion = nn.MSELoss()
        x_encoder = self.sigmoid(self.cs[-1])
        x_decoder = self.sigmoid(self.cs_rec_[-1]) #self.cs_decoder[-1])

        Le = criterion(x_encoder, x) * self.A * self.B
        Lf = criterion(x_decoder, x) * self.A * self.B

        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        Lz = torch.mean(Lz)  
        loss = La + Lb + Lz + Lx + Le + Lf
        return loss, imgs_rec


    # correct
    def filterbank_matrices(self,a,mu_x,sigma2,epsilon=1e-9):
      t_a,t_mu_x = align(a,mu_x)
      temp = t_a - t_mu_x
      temp,t_sigma = align(temp,sigma2)
      temp = temp / (t_sigma * 2)
      F = torch.exp(-torch.pow(temp,2))
      F = F / (F.sum(2,True).expand_as(F) + epsilon)
      return F

    def attn_window(self,h_enc):
      params = self.dec_linear(h_enc)
      gx_,gy_,log_sigma_2,log_delta,log_gamma = params.split(1,1)
      gx = (self.A + 1) / 2 * (gx_ + 1)
      gy = (self.B + 1) / 2 * (gy_ + 1)
      delta = (max(self.A,self.B) - 1) / (self.N - 1) * torch.exp(log_delta)
      sigma2 = torch.exp(log_sigma_2)
      gamma = torch.exp(log_gamma)

      return self.filterbank(gx,gy,sigma2,delta),gamma


    # correct
    def read(self,x,x_hat,h_enc_prev):
        (Fx,Fy),gamma = self.attn_window(h_enc_prev)
        def filter_img(img,Fx,Fy,gamma,A,B,N):
            Fxt = Fx.transpose(2,1)
            img = img.view(-1,B,A)
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1,N*N)
            return glimpse * gamma.view(-1,1).expand_as(glimpse)
        x = filter_img(x,Fx,Fy,gamma,self.A,self.B,self.N)
        x_hat = filter_img(x_hat,Fx,Fy,gamma,self.A,self.B,self.N)
        return torch.cat((x,x_hat),1)

    # correct
    def write(self,h_enc=0):
        w = self.enc_w_linear(h_enc)
        w = w.view(self.batch_size, self.N,self.N)
        (Fx,Fy),gamma = self.attn_window(h_enc)
        Fyt = Fy.transpose(2,1)
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size,self.A*self.B)
        return wr / gamma.view(-1,1).expand_as(wr)

    def sampleQ(self,h_enc):
        e = self.normalSample()
        mu = self.mu_linear(h_enc)
        log_sigma = self.sigma_linear(h_enc)
        sigma = torch.exp(log_sigma)
        return mu + sigma * e , mu , log_sigma, sigma

    def generate(self,batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size),volatile = True)
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size),volatile = True)

        for t in range(self.T):
            c_prev_decoder = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs_decoder[t - 1]
            s_t = self.normalSample()

            h_dec, dec_state = self.decoder(s_t, (h_dec_prev, dec_state))
            #v_t = self.write(h_dec)
            x_guess = self.sigmoid(self.generator(self.write(h_dec)))
            self.cs_d_[t] = x_guess
            h_dec_prev = h_dec

        imgs = []
        for img in self.cs_d_:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs
