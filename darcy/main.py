# From https://neuraloperator.github.io/neuraloperator/dev/auto_examples/plot_FNO_darcy.html

import neuralop
# import neuralop.wcw.tool_wcw as wcw
import torch
import math
import torch.fft as fft
from ml_model import model

from neuralop.datasets import load_darcy_flow_small

import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# train_loader, test_loaders, input_encoder,output_encoder = load_darcy_flow_small(
train_loader, test_loaders,output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
test_samples = test_loaders[32].dataset



# # wcw.check_dataloader(test_loaders[32])#Batch 0 : x_batch shape: torch.Size([32, 3, 32, 32]), y_batch shape: torch.Size([32, 1, 32, 32])
# 
# 
# 
# fig = plt.figure(figsize=(7, 7))
# for index in range(3):
#     data = test_samples[index]#dict x,y no batch
# 
#     # data = data_processor.preprocess(data, batched=False)
# 
#     # data['x']=input_encoder.encode(data['x'])
#     wcw.sss(data['x'])
# 
#     # Input x
#     x = data['x']#3,32,32
#     # Ground-truth
#     y = data['y']
#     # Model prediction
# 
#     out = model(x.unsqueeze(0))
#     out=output_encoder.encode(out)
# 
# 
#     ax = fig.add_subplot(3, 3, index*3 + 1)
#     ax.imshow(x[0], cmap='gray')
#     if index == 0:
#         ax.set_title('Input x')
#     plt.xticks([], [])
#     plt.yticks([], [])
# 
#     ax = fig.add_subplot(3, 3, index*3 + 2)
#     ax.imshow(y.squeeze())
#     if index == 0:
#         ax.set_title('Ground-truth y')
#     plt.xticks([], [])
#     plt.yticks([], [])
# 
#     ax = fig.add_subplot(3, 3, index*3 + 3)
#     ax.imshow(out.squeeze().detach().numpy())
#     if index == 0:
#         ax.set_title('Model prediction')
#     plt.xticks([], [])
#     plt.yticks([], [])
# 
# fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
# plt.tight_layout()
# plt.savefig('121.png')


class GaussianRF2d(object):

    def __init__(self, s1, s2, L1=1 , L2=1, alpha=2.0, tau=3.0, sigma=None, mean=None,
                 boundary="periodic", device=device, dtype=torch.float32):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau ** (0.5 * (2 * alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4 * (math.pi ** 2)) / (L1 ** 2)
        const2 = (4 * (math.pi ** 2)) / (L2 ** 2)

        # freq_list1 = torch.cat((torch.arange(start=0, end=s1 // 2, step=1),torch.arange(start=-s1 // 2, end=0, step=1)), 0)
        freq_list1=torch.arange(start=0,end=s1//2,step=1)
        k1 = freq_list1.view(-1, 1).repeat(1, s2 // 2 ).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2 // 2 , step=1)

        k2 = freq_list2.view(1, -1).repeat(s1//2, 1).type(dtype).to(device)

        self.sqrt_eig = self.sigma * ((const1 * k1 ** 2 + const2 * k2 ** 2 + tau ** 2) ** (-alpha / 2.0))
        # self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eiggg = self.sqrt_eig.clone()
        self.sqrt_eig[0, 0] = 0.0
        self.sqrt_eiggg[0, 0] = 1.0


    def sample(self, N, xi=None):
        if xi is None:
            xi = torch.randn(N, self.s1//2, self.s2 // 2, dtype=self.dtype, device=self.device)
            # xi = torch.randn(N, self.s1, self.s2 // 2 + 1, 2, dtype=self.dtype, device=self.device)

        xi[...] = self.sqrt_eig * xi[...]
        xi=torch.cat([xi,-xi],dim=1)
        xi=torch.cat([xi,-xi],dim=2)
        # xi[..., 1] = self.sqrt_eig * xi[..., 1]
        # print('xi',xi.shape)
        # print("xi_cplx",torch.view_as_complex(xi).shape)
        u = torch.real(fft.ifft2(xi, s=(self.s1, self.s2)))
        # print('u', u.shape)
        if self.mean is not None:
            u += self.mean

        return u
    def dist(self,u,v,eta=60):#N,x,y
        w=fft.fft2(u-v,norm='forward')
        residu=w[:,:self.s1//2,:self.s2//2]/self.sqrt_eiggg/eta
        norm_squared = torch.sum(residu.real ** 2 + residu.imag ** 2, dim=(1, 2))
        return norm_squared

grf = GaussianRF2d(32,32,1,1,alpha=1,tau=3.0,sigma=None)#,device=device,dtype=dtype)
nsum=10
random_gen=grf.sample(nsum)


# wcw.sss(random_gen)


fig = plt.figure(figsize=(7, 7))
data_y=test_samples[0]['y']
index=0
# ax = fig.add_subplot(3, 3, index*3 + 1)
# ax.imshow(x[0], cmap='gray')
# if index == 0:
#     ax.set_title('Input x')
# plt.xticks([], [])
# plt.yticks([], [])

ax = fig.add_subplot(3, 3, index*3 + 1)
ax.imshow(data_y.squeeze())
if index == 0:
    ax.set_title('Ground-truth y')
plt.xticks([], [])
plt.yticks([], [])

ax = fig.add_subplot(3, 3, index*3 + 2)
ax.imshow(data_y.squeeze()+60*random_gen[0].cpu())
# ax.imshow(out.squeeze().detach().numpy())
if index == 0:
    ax.set_title('Add Noice_1')
plt.xticks([], [])
plt.yticks([], [])

ax = fig.add_subplot(3, 3, index*3 + 3)
ax.imshow(data_y.squeeze()+60*random_gen[1].cpu())
# ax.imshow(out.squeeze().detach().numpy())
if index == 0:
    ax.set_title('Add Noice_2')
plt.xticks([], [])
plt.yticks([], [])

fig.suptitle('Add Noice.', y=0.98)
plt.tight_layout()
plt.savefig('add_noice.png')




import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm


n_post_sp=400
posit_embd=test_samples[0]['x'][1:].to(device)#2,32,32

y_noisy=test_samples[0]['y'].squeeze().to(device)+60*random_gen[0]#32,32
posit_embd=posit_embd.unsqueeze(0).repeat(n_post_sp,1,1,1)


def density1(k):#k0:3,32,32; k: n,x,y
    prior=-torch.sum(k**2,dim=(1,2))/2

    x=torch.cat([k.unsqueeze(1),posit_embd],dim=1)
    out=output_encoder.encode(model(x)).squeeze(1)
    distnorm=grf.dist(u=out,v=y_noisy)
    return -distnorm-prior



    # z = np.reshape(z, [z.shape[0], 2])
    # z1, z2 = z[:, 0], z[:, 1]
    # norm = np.sqrt(z1 ** 2 + z2 ** 2)
    # exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    # exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    # u = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)
    # return np.exp(-u)

def metropolis_hastings(target_density, size=100,N=n_post_sp):
    burnin_size = 25000
    size += burnin_size
    x0=torch.randn(N,32,32).to(device)
    # x0 = np.array([[0, 0]])
    xt = x0
    samples = []
    for i in tqdm(range(size)):
        # xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(2))])
        xt_candidate = xt+torch.randn(N,32,32).to(device)*0.1
        # accept_prob = (target_density(xt_candidate))/(target_density(xt))
        accept_prob = (target_density(xt_candidate))-(target_density(xt))

        random_j=torch.rand(n_post_sp).to(device)
        condition=random_j>torch.exp(accept_prob)
        xt[condition]=xt_candidate[condition]

        # if np.random.uniform(0, 1) < torch.exp(accept_prob):
        #     xt = xt_candidate
        samples.append(xt)
        if i%20==0:
            print(i)
    # samples.append(xt)
    samples=torch.cat(samples[burnin_size:],dim=0)
    # samples = np.array(samples[burnin_size:])
    # samples = np.reshape(samples, [samples.shape[0], 2])
    return samples
samples = metropolis_hastings(density1)
predict=torch.mean(samples[-n_post_sp:],dim=0).cpu().numpy()
# wcw.sss(predict)


fig = plt.figure(figsize=(7, 7))
for index in range(1):
    data = test_samples[index]#dict x,y no batch

    # data = data_processor.preprocess(data, batched=False)

    # data['x']=input_encoder.encode(data['x'])
    # wcw.sss(data['x'])

    # Input x
    x = data['x']#3,32,32
    # Ground-truth
    y = data['y']
    # Model prediction



    ax = fig.add_subplot(2, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(2, 3, index*3 + 3)
    ax.imshow(y_noisy.cpu())
    if index == 0:
        ax.set_title('Noisy Observation')
    plt.xticks([], [])
    plt.yticks([], [])
index=0
ax = fig.add_subplot(2, 3, index*3 + 4)
ax.imshow(predict)
if index == 0:
    ax.set_title('Inverse Prediction_average')
plt.xticks([], [])
plt.yticks([], [])

ax = fig.add_subplot(2, 3, index*3 + 5)
ax.imshow(samples[-1].cpu().numpy())
if index == 0:
    ax.set_title('Inverse Prediction_single')
plt.xticks([], [])
plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.savefig('pred.png')