import  numpy                       as      np
import  pandas                      as      pd
import  itertools
import  random
import  matplotlib.pyplot           as      plt
import  seaborn                     as      sns
import matplotlib.font_manager      as      fm
import matplotlib                   as      mpl

import  torch

import  os
plt.rcParams['axes.labelsize'] = 40
plt.rcParams.update({'font.size': 40})

#---
from    A001_Others.B004zC002_BackTransform    import  *
from    A001_Others.C008_Utils_ReplaceNames     import  *

###===>>>
def ExecuteB005(wgan_gp,
                 data_types,
                 All_Trainable_Data, df_fake,
                 Hyper002_Epochs
                 ):

    CurFolder = './Z000_Image1/'+'Epoch_'+str(Hyper002_Epochs)+'/'
    if not os.path.exists(CurFolder):
        os.mkdir(CurFolder)

    ###===###
    Real_Data = All_Trainable_Data.view(-1, 18)

    ###===###
    Real_Data = Execute_C002(Real_Data)
    Fake_Data = torch.tensor(df_fake.values)

    if Real_Data.shape[0] < Fake_Data.shape[0]:
        Sampled_Points  = \
            random.sample(range(Fake_Data.shape[0]), Real_Data.shape[0])
        Fake_Data = Fake_Data[Sampled_Points]

    if Fake_Data.shape[0] < Real_Data.shape[0]:
        Sampled_Points  = \
            random.sample(range(Real_Data.shape[0]), Fake_Data.shape[0])
        Real_Data = Real_Data[Sampled_Points] 

    ###===###
    Replace_Names = Execute_C008(data_types)

    #---
    ncols = 4
    for itr in range(16):
        if (itr == 0) or (itr == 10):
            fig, ax = plt.subplots(ncols = ncols, nrows = 5, figsize = (45, 40))

        if itr < 10:
            i, j = itr // ncols, itr - (itr // ncols) * ncols
        else:
            itr2 = itr - 10
            i, j = itr2 // ncols, itr2 - (itr2 // ncols) * ncols
        #---
        CurName = Replace_Names[itr]

        #---
        Cur_Fake = Fake_Data[:, itr].view(-1).cpu().detach()
        Cur_Real = Real_Data[:, itr].view(-1).cpu()

        #---
        df_fake = pd.DataFrame()
        df_fake[CurName] = Cur_Fake
        df_fake["Type"]  = "Synthetic"
        df_real = pd.DataFrame()
        df_real[CurName] = Cur_Real
        df_real["Type"]  = "Real"
        df_all = pd.concat([df_fake, df_real], ignore_index = True)

        #---
        plot = sns.kdeplot(
            x = df_all[CurName].astype(float).values,
            hue = df_all["Type"],
            fill = True,
            ax = ax[i, j]
            )
        plot.legend_.set_title(None)
        ax[i, j].yaxis.set_visible(False)
        ax[i, j].set(xlabel = CurName)
        plt.tight_layout(pad = 3.0)
        if (itr == 9):
            plt.savefig(CurFolder + str(1).zfill(3) +
                        'distribution_comparison_floats_part01' +
                        '.png')
            plt.close()

        if (itr == 15):
            fig.delaxes(ax[4, 1])
            fig.delaxes(ax[4, 2])
            fig.delaxes(ax[4, 3])
            plt.savefig(CurFolder + str(1).zfill(3) +
                        'distribution_comparison_floats_part02' +
                        '.png')
            plt.close()
         
    #---
    ReadFrom        = "./Z001_Data1/BTS/"
    BST_nonFloat    = torch.load(ReadFrom + "A001_BTS_nonFloat")

    ncols = 2
    fig, ax = plt.subplots(ncols = ncols, nrows = 1, figsize = (45, 24))
    for itr in range(16, Fake_Data.shape[1]):
        #---
        itr2 = itr - 16
        i, j = itr2 // ncols, itr2 - (itr2 // ncols) * ncols

        #---
        CurName = Replace_Names[itr]
        
        Cur_Fake = Fake_Data[:, itr].long().view(-1).cpu().detach().numpy()
        Cur_Real = Real_Data[:, itr].long().view(-1).cpu().numpy()

        #---
        df_fake = pd.DataFrame()
        df_fake[CurName] = Cur_Fake
        df_fake["Type"]  = "Synthetic"
        df_real = pd.DataFrame()
        df_real[CurName] = Cur_Real
        df_real["Type"]  = "Real"

        #---
        if itr <= 18:

            if itr == 16:
                mapper = {"0":"Male", "1":"Female"}
            else:
                mapper = {"0":"False", "1":"True"}
    
        #---
        df_all = pd.concat([df_fake, df_real], ignore_index = True)

        df_all[CurName] = df_all[CurName].astype(str).map(mapper)

        #---
        plot = sns.histplot(
                x = df_all[CurName],
                hue = df_all["Type"],
                fill = True,
                multiple = "dodge",
                shrink=0.8,
                alpha = 0.3,
                linewidth = 0,
                ax = ax[i]
                )

        plot.legend_.set_title(None)
        ax[i].yaxis.set_visible(False)
        ax[i].set(xlabel = CurName)
        plt.tight_layout(pad = 3.0)

    plt.savefig(CurFolder + str(2).zfill(3) +
                'distribution_comparison_nonfloat' +
                '.png')
    plt.close()

    #---
    name_replace = [i[8:] for i in list(data_types["name"])]
    DF_Real = pd.DataFrame(data = Real_Data.detach().numpy(),
                           columns = name_replace)
    DF_Fake = pd.DataFrame(data = Fake_Data.cpu().detach().numpy(),
                           columns = name_replace)

    real_matrix = DF_Real.astype(float).corr()
    fake_matrix = DF_Fake.astype(float).corr()

    mask = np.triu(np.ones_like(real_matrix, dtype = bool))

    fig, ax = plt.subplots(ncols = 1, nrows = 2, figsize = (25, 40))
    with sns.axes_style("white"):
        for i, (matrix, data_type) in enumerate(
            zip([fake_matrix, real_matrix], ["Synthetic", "Real"])
            ):
            sns.heatmap(
                matrix,
                cmap = "coolwarm",
                mask = mask,
                vmin = "-1",
                vmax = "1",
                linewidths = 0.5,
                square = True,
                ax = ax[i],
                ).set_title("Correlation Matrix: " + data_type + "Data")

    fig.tight_layout(pad = 3.0)

    plt.savefig(CurFolder + str(3).zfill(3) +
                'correlation_comparison' +
                '.png')
    plt.close()
        












