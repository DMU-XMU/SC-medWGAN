import  numpy               as np

import  torch
import  torch.nn            as  nn
import  torch.nn.functional as  F


#<<<===>>>
class MyLSTM(nn.Module):
    def __init__(self, ID, HD):
        super().__init__()
        self.ID = ID
        self.HD = HD
        self.i2h = nn.Linear(ID, HD * 4)
        self.h2h = nn.Linear(HD, HD * 4)

    def forward(self, x0):
        if x0.ndim == 2:  
            x0 = x0.unsqueeze(1)

        Q_k = torch.zeros(x0.shape[0], self.HD, device=x0.device)

        S_k = torch.zeros(x0.shape[0], self.HD, device=x0.device)
        Q_all = []

        for QStr in range(x0.shape[1]):
            X_k = x0[:, QStr, :]
            gates = self.i2h(X_k) + self.h2h(Q_k)
            F_k, I_k, A_k, O_k = gates.chunk(4, 1)

            F_k = torch.sigmoid(F_k)
            I_k = torch.sigmoid(I_k)
            A_k = torch.tanh(A_k)

            O_k = torch.sigmoid(O_k)

            S_k = F_k * S_k + I_k * A_k
            Q_k = O_k * torch.tanh(S_k)
            Q_all.append(Q_k.unsqueeze(1))

        Q_all = torch.cat(Q_all, dim=1)
        return Q_all, (Q_k, S_k)


#<<<===>>>
class Generator(nn.Module):

    ###===>>>
    def __init__(self,
                 Hyper006_ID, Hyper007_HD,
                 data_types):
        super().__init__()

        #---
        ID, HD = Hyper006_ID, Hyper007_HD

        #---
        self.rnn_f = MyLSTM(ID, HD)
        self.rnn_r = MyLSTM(ID, HD)

        self.linear1 = nn.Linear(2 * HD, HD)
        self.linear2 = nn.Linear(    HD, HD)

        self.linear3 = nn.Linear(HD, max(data_types["index_end"]))
        self.leakyReLU = nn.LeakyReLU(0.1)

        #---
        self.output_activations = []

        max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])
        self.output_activations.append(lambda x: torch.sigmoid(x[..., 0:max_real]))

        for index, row in data_types.iterrows():
            if row["type"] != "real":
                idxs = row["index_start"]
                idxe = row["index_end"]

                self.output_activations.append(
                    lambda x, idxs=idxs, idxe=idxe: torch.softmax(
                        x[..., idxs:idxe], dim=-1
                    )
                )

    ###===>>>
    def forward(self, x0):
        
        #---
        x0_f = x0
        x0_r = x0.flip(dims = [1])

        #---
        x1_f, _ = self.rnn_f(x0_f)
        x1_r, _ = self.rnn_r(x0_r)
        x1 = torch.cat((x1_f, x1_r), dim = 2)

        #---
        x2 = self.leakyReLU(self.linear1(x1))

        x3 = self.leakyReLU(self.linear2(x2))
        x4 = self.linear3(x3)

        #---
        x_list = [f(x4) for f in self.output_activations]
        out = torch.cat(x_list, dim=-1)

        ###===>>>
        return out

class Discriminator(nn.Module):
    def __init__(self, HD, data_types):
        super(Discriminator, self).__init__()
        self.max_real = max(data_types.loc[data_types["type"] == "real", "index_end"])
        self.embeddings = nn.ModuleList()
        self.embedding_slices = []

        total_embedding_size = 0
        for index, row in data_types.iterrows():
            if row['type'] != 'real':
                num_classes = row['num_classes']
                embedding_size = row['embedding_size']

                self.embeddings.append(nn.Embedding(num_classes, embedding_size))
                self.embedding_slices.append((row['index_start'], row['index_end'], embedding_size))
                total_embedding_size += embedding_size

        self.total_embedding_size = total_embedding_size  
        self.linear1 = nn.Linear(1, HD) 
        self.linear2 = nn.Linear(HD, HD)
        
        self.rnn_f = MyLSTM(HD, HD)
        self.rnn_r = MyLSTM(HD, HD)
        self.linear3 = nn.Linear(2 * HD, 1)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x0):
        if isinstance(x0, tuple):
            x0 = x0[0]
        # print(f"x0 shape: {x0.shape}")
        real_part = x0[..., :self.max_real]
        # print(f"real_part shape: {real_part.shape}")

        embeddings = [
            emb(x0[..., start:end].long()).view(x0.shape[0], x0.shape[1], -1)
            for (start, end, _), emb in zip(self.embedding_slices, self.embeddings)
        ]
        embeddings = torch.cat(embeddings, dim=-1)
        combined = torch.cat([real_part, embeddings], dim=-1)  


        if self.linear1.in_features != combined.shape[-1]:
            self.linear1 = nn.Linear(combined.shape[-1], self.linear1.out_features).to(combined.device)
            # print(f"Adjusted linear1 input dimension to: {combined.shape[-1]}")

        x1 = self.leakyReLU(self.linear1(combined))
        x2 = self.leakyReLU(self.linear2(x1))

        # 修复：只取 LSTM 输出的第一部分
        x2_f, _ = self.rnn_f(x2)
        x2_r, _ = self.rnn_r(torch.flip(x2, dims=[1]))
        x3 = torch.cat((x2_f, x2_r), dim=-1)

        output = self.linear3(x3)
        return output
