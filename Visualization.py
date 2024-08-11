'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

PCA and TSNE analysis between Original data and Synthetic data
Inputs: 
  - dataX: original data
  - dataX_hat: synthetic data
  
Outputs:
  - PCA Analysis Results
  - t-SNE Analysis Results

'''
# %% Necessary Packages

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据导入
original_data_path = 'xxxx'

generated_data_path = 'xxxx'

original_df = pd.read_csv(original_data_path)


generated_df = pd.read_csv(generated_data_path)

original_df.columns = original_df.columns.astype(str)

generated_df.columns = generated_df.columns.astype(str)

original_df = original_df.iloc[:,1:]
generated_df = generated_df.iloc[:,1:]


# %% TSNE Analysis

def tSNE_Analysis(dataX, dataX_hat):
    # Analysis Data Size
    Sample_No = 1900
    

    # Preprocess
    # for i in range(Sample_No):
    #     if (i == 0):
    #         arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
    #         arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
    #     else:
    #         arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
    #         arrayX_hat = np.concatenate(
    #             (arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])])))

    # Do t-SNE Analysis together       
    #final_arrayX = np.concatenate((arrayX, arrayX_hat), axis=0)

    columns = list(set(dataX.columns) & set(dataX_hat.columns))
    final_df = pd.concat([dataX.loc[:Sample_No-1,columns], dataX_hat.loc[:Sample_No-1,columns]],ignore_index=True)

    # Parameters
  
    colors = ["red" for i in range(Sample_No)] + ["blue" for i in range(Sample_No)]
    print(len(colors))
    # TSNE anlaysis
    tsne = TSNE(n_components=2,random_state=42,verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(final_df)

    # Plotting
    f, ax = plt.subplots(1)
    print(tsne_results.shape)
    plt.scatter(tsne_results[:Sample_No, 0], tsne_results[:Sample_No, 1], c=colors[:Sample_No], alpha=0.2, label="Original")
    plt.scatter(tsne_results[Sample_No:, 0], tsne_results[Sample_No:, 1], c=colors[Sample_No:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig("xxxx.jpg", figsize=[50, 43])
    plt.show()


tSNE_Analysis(original_df, generated_df)
