
# Plots to illustrate SLI and mixup results by the number of pairs, for beta lambda (largesample)

import matplotlib.pyplot as plt
import numpy as np

numpairs = np.array([560, 1125, 2250, 4500, 5625, 11250, 22500, 45000])
numpairsmixup = np.array([1125, 4500, 22500])
ecelargesample = df_ece.iloc[51:59, 1]
mcelargesample = df_mce.iloc[51:59, 1]
losslargesample = df_loss.iloc[51:59, 1]
errorlargesample = df_error.iloc[51:59, 1]
ecelargesamplemixup = df_ece.iloc[59:62, 1]
mcelargesamplemixup = df_mce.iloc[59:62, 1]
losslargesamplemixup = df_loss.iloc[59:62, 1]
errorlargesamplemixup = df_error.iloc[59:62, 1]

# style
plt.style.use('seaborn-whitegrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

# Plot ECE 
plt.figure(1)
plt.plot(numpairs, ecelargesample, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Latent Blending')

# Add legend and labels
#plt.legend(loc='best')
plt.title("ECE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("ECE")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/largesampleecebypairs.png')

# Plot MCE 
plt.figure(2)
plt.plot(numpairs, mcelargesample, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Latent Blending')

# Add legend and labels
#plt.legend(loc='best')
plt.title("MCE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("MCE")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/largesamplemcebypairs.png')

# Plot error
plt.figure(3)
plt.plot(numpairs, errorlargesample, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Latent Blending')

# Add legend and labels
#plt.legend(loc='best')
plt.title("Error by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Error (%)")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/largesampleerrorbypairs.png')

# Plot loss 
plt.figure(4)
plt.plot(numpairs, losslargesample, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Latent Blending')

# Add legend and labels
#plt.legend(loc='best')
plt.title("Loss by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Loss")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/largesamplelossbypairs.png')

# Plots to illustrate SLI results against the baseline by number of pairs, for uniform lambda

import matplotlib.pyplot as plt
import numpy as np

numpairs = np.array([560, 1125, 2250, 4500])

# style
plt.style.use('seaborn-whitegrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

# Plot ECE for mixup by alpha value
eceuniformaugmean1 = df_ece.iloc[1:5, 1]
eceuniformaugmean2 = df_ece.iloc[9:13, 1]
eceuniformaugmean3 = df_ece.iloc[17:21, 1]
eceuniformaugmean4 = df_ece.iloc[25:29, 1]

plt.figure(1)
plt.plot(numpairs, eceuniformaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, eceuniformaugmean2, markerfacecolor='blue', markersize=12, color=palette(2), linewidth=4, label='SLI-CP')
plt.plot(numpairs, eceuniformaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')
plt.plot(numpairs, eceuniformaugmean4, markerfacecolor='blue', markersize=12, color=palette(4), linewidth=4, label='Slerp-CP')

# Add legend and labels
plt.legend(loc='best')
plt.title("ECE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("ECE")
plt.xlim(500, 4500)
plt.ylim(0.009, 0.017)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/uniformecebypairs.png')

# Plot MCE for mixup by alpha value
mceuniformaugmean1 = df_mce.iloc[1:5, 1]
mceuniformaugmean2 = df_mce.iloc[9:13, 1]
mceuniformaugmean3 = df_mce.iloc[17:21, 1]
mceuniformaugmean4 = df_mce.iloc[25:29, 1]

plt.figure(2)
plt.plot(numpairs, mceuniformaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, mceuniformaugmean2, markerfacecolor='blue', markersize=12, color=palette(2), linewidth=4, label='SLI-CP')
plt.plot(numpairs, mceuniformaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')
plt.plot(numpairs, mceuniformaugmean4, markerfacecolor='blue', markersize=12, color=palette(4), linewidth=4, label='Slerp-CP')

# Add legend and labels
#plt.legend(loc='best')
plt.title("MCE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("MCE")
plt.xlim(500, 4500)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/uniformmcebypairs.png')

# Plot Error for mixup by alpha value
erroruniformaugmean1 = df_error.iloc[1:5, 1]
erroruniformaugmean2 = df_error.iloc[9:13, 1]
erroruniformaugmean3 = df_error.iloc[17:21, 1]
erroruniformaugmean4 = df_error.iloc[25:29, 1]

plt.figure(3)
plt.plot(numpairs, erroruniformaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, erroruniformaugmean2, markerfacecolor='blue', markersize=12, color=palette(2), linewidth=4, label='SLI-CP')
plt.plot(numpairs, erroruniformaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')
plt.plot(numpairs, erroruniformaugmean4, markerfacecolor='blue', markersize=12, color=palette(4), linewidth=4, label='Slerp-CP')

# Add legend and labels
plt.legend(loc='best')
plt.title("Error by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Error")
plt.xlim(500, 4500)
plt.ylim(6.8,7.6)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/uniformerrorbypairs.png')

# Plot Loss for mixup by alpha value
lossuniformaugmean1 = df_loss.iloc[1:5, 1]
lossuniformaugmean2 = df_loss.iloc[9:13, 1]
lossuniformaugmean3 = df_loss.iloc[17:21, 1]
lossuniformaugmean4 = df_loss.iloc[25:29, 1]

plt.figure(4)
plt.plot(numpairs, lossuniformaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, lossuniformaugmean2, markerfacecolor='blue', markersize=12, color=palette(2), linewidth=4, label='SLI-CP')
plt.plot(numpairs, lossuniformaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')
plt.plot(numpairs, lossuniformaugmean4, markerfacecolor='blue', markersize=12, color=palette(4), linewidth=4, label='Slerp-CP')

# Add legend and labels
plt.legend(loc='best')
plt.title("Loss by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Loss")
plt.xlim(500, 4500)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/uniformlossbypairs.png')

# Plots to illustrate SLI results against the baseline by number of pairs, for beta lambda

import matplotlib.pyplot as plt
import numpy as np

numpairs = np.array([560, 1125, 2250, 4500])

# style
plt.style.use('seaborn-whitegrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

# Plot ECE for mixup by alpha value
ecebetaaugmean1 = df_ece.iloc[43:47, 1]
ecebetaaugmean3 = df_ece.iloc[47:51, 1]

plt.figure(1)
plt.plot(numpairs, ecebetaaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, ecebetaaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')

# Add legend and labels
plt.legend(loc='best')
plt.title("ECE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("ECE")
plt.xlim(500,4500)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/betaecebypairs.png')

# Plot MCE for mixup by alpha value
mcebetaaugmean1 = df_mce.iloc[43:47, 1]
mcebetaaugmean3 = df_mce.iloc[47:51, 1]

plt.figure(2)
plt.plot(numpairs, mcebetaaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, mcebetaaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')

# Add legend and labels
plt.legend(loc='best')
plt.title("MCE by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("MCE")
plt.xlim(500,4500)
plt.ylim(0.06,0.25)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/betamcebypairs.png')

# Plot Error for mixup by alpha value
errorbetaaugmean1 = df_error.iloc[43:47, 1]
errorbetaaugmean3 = df_error.iloc[47:51, 1]

plt.figure(3)
plt.plot(numpairs, errorbetaaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, errorbetaaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')

# Add legend and labels
plt.legend(loc='best')
plt.title("Error by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Error")
plt.xlim(500,4500)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/betaerrorbypairs.png')

# Plot loss for mixup by alpha value
lossbetaaugmean1 = df_loss.iloc[43:47, 1]
lossbetaaugmean3 = df_loss.iloc[47:51, 1]

plt.figure(4)
plt.plot(numpairs, lossbetaaugmean1, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='SLI')
plt.plot(numpairs, lossbetaaugmean3, markerfacecolor='blue', markersize=12, color=palette(3), linewidth=4, label='Slerp')

# Add legend and labels
plt.legend(loc='best')
plt.title("Loss by No. of Pairs", fontsize=12, fontweight=0, color='black')
plt.xlabel("No. of Pairs")
plt.ylabel("Loss")
plt.xlim(500,4500)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/betalossbypairs.png')

# Plots to illustrate results for mixup for various values of lambda

import matplotlib.pyplot as plt
import numpy as np

alphas = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
numpairs = np.array([560, 1125, 2250, 4500, 5625, 11250, 22500, 45000, 90000])
ecemixup = df_ece.iloc[33:39, 1]
mcemixup = df_mce.iloc[33:39, 1]
lossmixup = df_loss.iloc[33:39, 1]
errormixup = df_error.iloc[33:39, 1]

# style
plt.style.use('seaborn-whitegrid')
 
# create a color palette
palette = plt.get_cmap('Set1')

# Plot ECE for mixup by alpha value
plt.figure(1)
plt.plot(alphas, ecemixup, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='ECE')
plt.xlim(0.0, 1.0)
plt.ylim(0.0077, 0.013)

# Add legend and labels
#plt.legend(loc='best')
plt.title("ECE by Alpha, Mixup", fontsize=12, fontweight=0, color='black')
plt.xlabel("Alpha")
plt.ylabel("ECE")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/mixupecebyalpha.png')

# Plot MCE for mixup by alpha value
plt.figure(2)
plt.plot(alphas, mcemixup, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='MCE')
plt.xlim(0.0, 1.0)
plt.plot(alphas, mcemixup, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='MCE')
plt.ylim(0.08, 0.35)

# Add legend and labels
#plt.legend(loc='best')
plt.title("MCE by Alpha, Mixup", fontsize=12, fontweight=0, color='black')
plt.xlabel("Alpha")
plt.ylabel("MCE")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/mixupmcebyalpha.png')

# Plot error for mixup by alpha value
plt.figure(3)
plt.plot(alphas, errormixup, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Error')


# Add legend and labels
#plt.legend(loc='best')
plt.title("Error by Alpha, Mixup", fontsize=12, fontweight=0, color='black')
plt.xlabel("Alpha")
plt.ylabel("Error (%)")
plt.xlim(0.0, 1.0)

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/mixuperrorbyalpha.png')

# Plot loss for mixup by alpha value
plt.figure(4)
plt.plot(alphas, lossmixup, markerfacecolor='blue', markersize=12, color=palette(1), linewidth=4, label='Loss')
plt.xlim(0.0, 1.0)

# Add legend and labels
#plt.legend(loc='best')
plt.title("Loss by Alpha, Mixup", fontsize=12, fontweight=0, color='black')
plt.xlabel("Alpha")
plt.ylabel("Loss")

# Plot
plt.savefig('/Users/wildflowerlyi/Desktop/Plots/mixuplossbyalpha.png')
