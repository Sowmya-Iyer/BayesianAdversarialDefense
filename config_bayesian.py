############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
# priors={
#     'prior_mu': 0,
#     'prior_sigma': 0.1,
#     'posterior_mu_initial': (-2, 0.1),  # (mean, std) normal_
#     'posterior_rho_initial': (-2, 0.1),  # (mean, std) normal_
# }
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 100
lr_start = 0.01
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 'Blundell'  # 'Blundell', 'Standard', Use float for const value
print("Parameters initialized")