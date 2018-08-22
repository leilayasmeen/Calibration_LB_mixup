
# This code was obtained from Markus Kangsepp's Github repository and adjusted for this paper.

import numpy as np
import pandas as pd
from os.path import join
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from cal_methods import TemperatureScaling, evaluate, softmax, cal_results

PATH = '/Users/wildflowerlyi/Desktop/Github/NN_calibration/'
files = ('resnet_cifar/probs_resnet110_c10clip_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_2250mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_1125mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_560mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_2250mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_1125mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_560mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol2_560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_2250mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_1125mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_560mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol3_560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_2250mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_1125mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_560mean_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_interpol4_560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixupalpha1_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixupalpha06_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixupalpha04_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixupalpha01_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixupalpha005_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_varyprop_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_varyprop2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_varyprop1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_varyprop560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_2250mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_1125mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug_560mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug3mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug3_2250mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug3_1125mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_aug3_560mean_beta_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_45000_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_22500_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_11250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_5625_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_4500_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_augmean_beta_largesample_560_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_largesample_baseline_22500_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_11250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_5625_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_4500_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_2250_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_1125_logits.p'
         ,'resnet_cifar/probs_resnet110_c10clip_mixup_baseline_560_logits.p'
        )
        
# Dataframes with results after applying two different calibration methods
df_iso = cal_results(IsotonicRegression, PATH, files, {'y_min':0, 'y_max':1}, approach = "single")
df_temp_scale = cal_results(TemperatureScaling, PATH, files, approach = "all")
        
dfs = [df_iso, df_temp_scale]
        
names = ["Name", "Uncalibrated", "Isotonic Regression", "Temperature Scaling"]


def get_dataframe(dfs, column, names):

    df_res = pd.DataFrame(columns=names)

    for i in range(1, len(df_iso), 2):

        name = dfs[0].iloc[i-1]["Name"] # Get name of method
        uncalibrated = dfs[0].iloc[i-1][column]  # Get uncalibrated score

        row = [name, uncalibrated]  # Add scores to row

        for df in dfs:
            row.append(df.iloc[i][column])

        df_res.loc[(i-1)//2] = row
    
    df_res.set_index('Name', inplace = True)
        
    return df_res

df_error = get_dataframe(dfs, "Error", names)
df_ece = get_dataframe(dfs, "ECE", names)
df_mce = get_dataframe(dfs, "MCE", names)
df_loss = get_dataframe(dfs, "Loss", names)

def highlight_min(s):
    '''
    highlight the min in a Series yellow.
    '''
    is_max = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_max]
    
# Summary tables for all metrics

df_error.style.apply(highlight_min, axis = 1)
df_ece.style.apply(highlight_min, axis = 1)
df_mce.style.apply(highlight_min, axis = 1)
df_loss.style.apply(highlight_min, axis = 1)
		
