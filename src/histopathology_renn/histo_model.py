#import numpy as np
import torch
import torch.nn as nn

class lmm_nn(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.f = nn.Sequential(
      nn.Linear(input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )
    
    self.log_sigma2_e_est = nn.Parameter(torch.tensor(0.0))
    self.log_sigma2_b_est = nn.Parameter(torch.tensor(0.0))
  
  def forward(self, x):
    fx = self.f(x).squeeze()
    return fx
  
  #### UNADJUSTED NLL
  def nll(self, y, fx):
    sigma2_e_est = torch.exp(self.log_sigma2_e_est)
    sigma2_b_est = torch.exp(self.log_sigma2_b_est)

    n = y.shape[0]
    device = y.device

    #V = sigma2_b * 1 * 1' + sigma2_e * I
    ones_vect = torch.ones(n , 1, device=device)
    V = sigma2_b_est * (ones_vect @ ones_vect.T) + sigma2_e_est * torch.eye(n, device=device)

    logdet_V = (n-1) * torch.log(sigma2_e_est) + torch.log(sigma2_e_est + n * sigma2_b_est)
    V_inv = (1 / sigma2_e_est ) * torch.eye(n , device=device) - (sigma2_b_est / (sigma2_e_est * (sigma2_e_est + n * sigma2_b_est))) * (ones_vect @ ones_vect.T)
    
    e = (y - fx).unsqueeze(1) # shape : (n ,1)
    nll = 0.5 * logdet_V + 0.5 * e.T @ V_inv @ e + 0.5 * n * torch.log( torch.tensor( 2 * torch.pi, device=device))

    return nll.squeeze()
  
  ### ADJUSTED NLL Version 1
  def nll_cor_1(self, y, fx, n):
    #nll with Submatrix of Inverse V (not Invsere of Submatrix V_A) & correction for log[det(submatrix V)]
    sigma2_e_est = torch.exp(self.log_sigma2_e_est)
    sigma2_b_est = torch.exp(self.log_sigma2_b_est)

    
    device = y.device
    
    # V inverse based on full group size (n)
    #V = sigma2_b * 1 * 1' + sigma2_e * I
    ones_vect = torch.ones(n , 1, device=device)
    V = sigma2_b_est * (ones_vect @ ones_vect.T) + sigma2_e_est * torch.eye(n, device=device)
    V_inv = (1 / sigma2_e_est ) * torch.eye(n , device=device) - (sigma2_b_est / (sigma2_e_est * (sigma2_e_est + n * sigma2_b_est))) * (ones_vect @ ones_vect.T)
    
    m = y.shape[0] #minibatch size
    V_inv_sub = V_inv[0:m , 0:m]

    #log det of submatrix of V (use m)
    logdet_V = (m-1) * torch.log(sigma2_e_est) + torch.log(sigma2_e_est + m * sigma2_b_est)

    e = (y - fx).unsqueeze(1) # shape : (m ,1)

    #correction term distributes log det error proportional
    alpha = (m/n) *  0.5 * torch.log((sigma2_e_est * (sigma2_e_est + n * sigma2_b_est)) /
                                     ((sigma2_e_est + m * sigma2_b_est) * (sigma2_e_est + (n-m) * sigma2_b_est))
                                      )
    nll = 0.5 * logdet_V + 0.5 * e.T @ V_inv_sub @ e + 0.5 * m * torch.log( torch.tensor( 2 * torch.pi, device=device)) + alpha

    return nll.squeeze()
  
  #### MISSPECIFIED MODEL no RE
  def nll_no_re(self, y, fx):
    sigma2_e_est = torch.exp(self.log_sigma2_e_est)
    
    n = y.shape[0]
    device = y.device

    logdet_V = n * torch.log(sigma2_e_est)
    V_inv =  (1 / sigma2_e_est ) *  torch.eye(n, device=device)

    e = (y - fx ).unsqueeze(1) # shape: (n,)
    nll = 0.5 * e.T @ V_inv @ e + 0.5 * logdet_V + 0.5 * n * torch.log( torch.tensor( 2 * torch.pi, device=device))
    return nll.squeeze()
  
  ### NLL ADJUSTMENT version 2
  def nll_cor_det(self, y, fx, n):
    sigma2_e_est = torch.exp(self.log_sigma2_e_est)
    sigma2_b_est = torch.exp(self.log_sigma2_b_est)

    m = y.shape[0]
    device = y.device

    #V = sigma2_b * 1 * 1' + sigma2_e * I
    ones_vect = torch.ones(m , 1, device=device)
    V = sigma2_b_est * (ones_vect @ ones_vect.T) + sigma2_e_est * torch.eye(m, device=device)

    logdet_V = (m-1) * torch.log(sigma2_e_est) + torch.log(sigma2_e_est + m * sigma2_b_est)
    V_inv = (1 / sigma2_e_est ) * torch.eye(m , device=device) - (sigma2_b_est / (sigma2_e_est * (sigma2_e_est + m * sigma2_b_est))) * (ones_vect @ ones_vect.T)
    
    alpha = (m/n) *  0.5 * torch.log((sigma2_e_est * (sigma2_e_est + n * sigma2_b_est)) /
                                     ((sigma2_e_est + m * sigma2_b_est) * (sigma2_e_est + (n-m) * sigma2_b_est))
                                      )
    
    e = (y - fx).unsqueeze(1) # shape : (n ,1)
    nll = 0.5 * logdet_V + 0.5 * e.T @ V_inv @ e + 0.5 * m * torch.log( torch.tensor( 2 * torch.pi, device=device)) + alpha

    return nll.squeeze()

  def get_var_est(self):
    return torch.exp(self.log_sigma2_e_est).item(), torch.exp(self.log_sigma2_b_est).item()
  