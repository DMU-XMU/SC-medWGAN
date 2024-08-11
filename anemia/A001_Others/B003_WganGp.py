import  numpy                   as      np
import  pandas                  as      pd
import  itertools
import  random
import  yaml

import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
import  torch.optim         as      optim
from    torch               import  autograd

#---
from    A001_Others.B003zC001_Models    import *

###===>>>
def correlation(x, eps = 1e-8):
    last_dim = x.shape[-1]
    x = x.reshape((-1, last_dim))
    x = x - x.mean(dim = 0, keepdim = True)
    x = x / torch.clamp(x.norm(dim = 0, keepdim = True), min = eps)
    correlation_matrix = x.transpose(0, 1) @ x

    ###===>>>
    return correlation_matrix

###===>>>
def LoadPreTrain(content = [False, 'G_SD', 'D_SD', 0]):

    Continue = content[0]
    Load_From = './Z002_Parameters/Epoch_' + str(content[3]) + '/'
    
    G_SD = Load_From + content[1]
    D_SD = Load_From + content[2]

    if Continue:
        G_SD = torch.load(G_SD)
        D_SD = torch.load(D_SD)

        ###===>>>
        return G_SD, D_SD

    else:
        ###===>>>
        return 0, 0

#<<<===>>>
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 使用sigmoid确保输出在0到1之间
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class GeneratorState(nn.Module):
    def __init__(self, ID, HD, state_dim):
        super(GeneratorState, self).__init__()
        self.rnn = MyLSTM(ID, HD)
        self.linear = nn.Linear(HD, state_dim)
        self.activation = nn.Tanh()


        self.autoencoder = AutoEncoder(state_dim, latent_dim=10) 

    def forward(self, x):
        output, _ = self.rnn(x)
        state = self.activation(self.linear(output))

        # 使用自动编码器对state进行重构
        recon, latent = self.autoencoder(state)
        return recon, latent


class GeneratorDose(nn.Module):
    def __init__(self, state_dim, HD, dose_dim):
        super(GeneratorDose, self).__init__()
        self.rnn = MyLSTM(state_dim, HD)
        self.linear = nn.Linear(HD, dose_dim)
        self.activation = nn.Sigmoid()


    def forward(self, state):
        output, _ = self.rnn(state)
        dose = self.activation(self.linear(output))
        return dose

class ExecuteB003:


    def __init__(self, All_Trainable_Data, Hyper001_BatchSize, Hyper002_Epochs, Hyper003_G_iter, Hyper004_GP_Lambda, Hyper005_C_Lambda, Hyper006_ID, Hyper007_HD, Hyper008_LR, Hyper009_Betas, data_types, continue_info=[False, 'G_SD', 'D_SD', 0], state_dim=10, dose_dim=5):
        super().__init__()
        self.batch_size = Hyper001_BatchSize
        self.epochs = Hyper002_Epochs
        self.G_iter = Hyper003_G_iter
        self.gp_weight = Hyper004_GP_Lambda
        self.c_weight = Hyper005_C_Lambda
        self.ID = Hyper006_ID

        self.HD = Hyper007_HD
        self.lr = Hyper008_LR
        self.betas = Hyper009_Betas

        self.gen_state = GeneratorState(self.ID, self.HD, state_dim)
        self.gen_dose = GeneratorDose(state_dim, self.HD, dose_dim)

        if torch.cuda.is_available():
            self.CUDA = True
            self.gen_state.cuda()

            self.gen_dose.cuda()
        else:
            self.CUDA = False

        self.G = Generator(self.ID, self.HD, data_types)
        self.D = Discriminator(self.HD, data_types)

        if self.CUDA:
            self.G.cuda()
            self.D.cuda()

        G_SD, D_SD = LoadPreTrain(continue_info)
        if G_SD != 0:
            self.G.load_state_dict(G_SD)
            self.D.load_state_dict(D_SD)

            self.PreviousEpoch = continue_info[3]
        else:
            self.PreviousEpoch = 0

        self.D_opt = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.betas)
        self.G_opt = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.betas)

    def state_dose_correlation_loss(states, doses):
        mse = nn.MSELoss()
        loss = mse(states, doses)
        return loss

    def generate_data(self, seq_len, num_samples=None):
        if num_samples is None:
            num_samples = self.batch_size

        noise = torch.rand((num_samples, seq_len, self.ID)).cuda()
        states, _ = self.gen_state(noise)
        doses = self.gen_dose(states)

        generated_data = torch.cat((states, doses), dim=-1)
        generated_data = generated_data.view(num_samples, seq_len, -1)
        generated_data = F.pad(generated_data, (0, 104 - generated_data.shape[2]), "constant", 0)  
        return generated_data, noise  


    # 在训练迭代中添加梯度裁剪
    def _critic_train_iteration(self, data_real):
        data_real = data_real.cuda()
        data_fake, noise = self.generate_data(data_real.shape[1], data_real.shape[0])
        D_real = self.D(data_real)

        D_fake = self.D(data_fake)
        with torch.backends.cudnn.flags(enabled=False):
            gradient_penalty = self._gradient_penalty(data_real, data_fake)
        self.D_opt.zero_grad()
        D_loss = D_fake.mean() - D_real.mean() + gradient_penalty
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)  
        self.D_opt.step()
        return D_loss.item(), gradient_penalty.item()

    def _generator_train_iteration(self, seq_len):
        data_fake, noise = self.generate_data(seq_len)
        D_fake = self.D(data_fake)

        # 计算生成器损失
        G_loss = -D_fake.mean()

        # 添加特征损失
        correlation_loss = self._correlation_loss(data_fake)
        G_loss += self.c_weight * correlation_loss

        # 使用生成数据时的噪声来计算自动编码器重构损失
        recon, _ = self.gen_state(noise)
        # 裁剪或填充 recon 使其与 noise 的尺寸匹配
        if recon.size(-1) > noise.size(-1):
            recon = recon[..., :noise.size(-1)]  # 裁剪 recon
        else:
            # 填充 recon 至 noise 的尺寸
            pad_size = noise.size(-1) - recon.size(-1)
            recon = F.pad(recon, (0, pad_size), "constant", 0)

        ae_loss = F.mse_loss(recon, noise)

        G_loss += ae_loss

        self.G_opt.zero_grad()
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.G_opt.step()

        return G_loss.item()

    def _correlation_loss(self, data_fake):
        correlation_fake = correlation(data_fake)
        criterion = nn.L1Loss(reduction="mean")
        return criterion(correlation_fake, self.correlation_real)
    def _gradient_penalty(self, data_real, data_fake):
        # 确保 alpha 的形状和 data_real, data_fake 一致
        alpha = torch.rand((self.batch_size, 1, 1), device=data_real.device)
        alpha = alpha.expand_as(data_real)

        # 确保数据类型都是 float 类型
        data_real = data_real.float()
        data_fake = data_fake.float()

        interpolated = alpha * data_real + (1 - alpha) * data_fake

        prob_interpolated = self.D(interpolated)

        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def train(self, All_loader):
        All_Length = list(All_loader.keys())
        All_Length.sort()

        first_batch = next(iter(All_loader[All_Length[0]]))
        data_real = first_batch[0].cuda()  
        
        self.correlation_real = correlation(data_real)

        for epoch in range(self.epochs - self.PreviousEpoch):
            for Ltr in range(len(All_Length)):
                Cur_Len = All_Length[Ltr]
                Cur_loader = All_loader[Cur_Len]
                print("###===>>>")

                for batch_idx, (data_real, _) in enumerate(Cur_loader):
                    data_real = data_real.cuda()
                    for itr in range(self.G_iter):
                        D_Loss, GP = self._critic_train_iteration(data_real)
                    G_Loss = self._generator_train_iteration(seq_len=Cur_Len)
                print("###===###")
                print("Epoch: \t{}".format(self.PreviousEpoch + epoch + 1))
                print("Loader Len: \t{}".format(Cur_Len))
                print("---" * 3)
                print("D_Loss: \t{}".format(D_Loss))
                print("GP: \t\t{}".format(GP))
                print("G_Loss: \t{}".format(G_Loss))
                print("")