# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 05:54:08 2025

@author: bailige
"""

import numpy as np
import CrossGradient_B as CG_B
import matplotlib.pyplot as plt

CG=CG_B.CrossGradient()
#%% SA_CG, SB_CG=CG.dx_dz(A, B, a_b='ab', ni, tol_ , smooth, coef_x, coef_y)
##  a_b= 'ab'   || A and B are similar to each other
##  a_b= 'ab'   || A is similar to B
##  a_b= 'ab'   || B is similar to A
##  a_b= 'No'   || Do not use cross gradient
##  smooth=True || Enable Smooth, with parameters coef_x and coef_x set to 2 by default
##  ni and tol_ || Iteration times and stability error
##  original author: diego domenzain  https://github.com/diegozain/alles
##  Copy author: Bai Lige  https://github.com/gebai7

#%% Initialize parameters
N_Sa=3600
y1=40
x1=90
re_data = np.loadtxt('shat4.txt', dtype=np.float64)

#%% Reform
reSant=re_data[:N_Sa]
rePhiT=re_data[N_Sa:]

CG_sa=reSant.reshape((y1),(x1))
CG_ph=rePhiT.reshape((y1),(x1))


#%% CG test
## A and B are similar to each other
SA_CG_AB, SB_CG_AB=CG.dx_dz(CG_sa, CG_ph,a_b='ab', ni = 200,tol_ = 1e-20,smooth=True,coef_x=2,coef_y=2)

## A is similar to B
SA_CG_A, SB_CG_A=CG.dx_dz(CG_sa, CG_ph,a_b='a', ni = 200,tol_ = 1e-20,smooth=True,coef_x=2,coef_y=2)

## B is similar to A
SA_CG_B, SB_CG_B=CG.dx_dz(CG_sa,CG_ph,a_b='b', ni = 200,tol_ = 1e-20,smooth=True,coef_x=2,coef_y=2)


#%% plot

fig = plt.figure(figsize=(12, 10))
ax1 = plt.subplot(4, 2, 1)
im1 = ax1.imshow(CG_sa, cmap='jet', aspect='auto', origin='upper', vmin=5, vmax=30)
ax1.set_title('True SA', fontsize=10)
ax1.set_xticks([]); ax1.set_yticks([])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Value', fontsize=8)

ax2 = plt.subplot(4, 2, 2)
im2 = ax2.imshow(CG_ph, cmap='jet', aspect='auto', origin='upper', vmin=20, vmax=40)
ax2.set_title('True PH', fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Intensity', fontsize=8)

ax1 = plt.subplot(4, 2, 3)
im1 = ax1.imshow(SA_CG_AB, cmap='jet', aspect='auto', origin='upper', vmin=5, vmax=30)
ax1.set_title('ab SA', fontsize=10)
ax1.set_xticks([]); ax1.set_yticks([])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Value', fontsize=8)

ax2 = plt.subplot(4, 2, 4)
im2 = ax2.imshow(SB_CG_AB, cmap='jet', aspect='auto', origin='upper', vmin=20, vmax=40)
ax2.set_title('ab PH', fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Intensity', fontsize=8)

ax1 = plt.subplot(4, 2, 5)
im1 = ax1.imshow(SA_CG_A, cmap='jet', aspect='auto', origin='upper', vmin=5, vmax=30)
ax1.set_title('a SA', fontsize=10)
ax1.set_xticks([]); ax1.set_yticks([])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Value', fontsize=8)

ax2 = plt.subplot(4, 2, 6)
im2 = ax2.imshow(SB_CG_A, cmap='jet', aspect='auto', origin='upper', vmin=20, vmax=40)
ax2.set_title('a PH', fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Intensity', fontsize=8)

ax1 = plt.subplot(4, 2, 7)
im1 = ax1.imshow(SA_CG_B, cmap='jet', aspect='auto', origin='upper', vmin=5, vmax=30)
ax1.set_title('B SA', fontsize=10)
ax1.set_xticks([]); ax1.set_yticks([])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Value', fontsize=8)

ax2 = plt.subplot(4, 2, 8)
im2 = ax2.imshow(SB_CG_B, cmap='jet', aspect='auto', origin='upper', vmin=20, vmax=40)
ax2.set_title('B PH', fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Intensity', fontsize=8)

plt.savefig('subplots.png', dpi=300, bbox_inches='tight')
