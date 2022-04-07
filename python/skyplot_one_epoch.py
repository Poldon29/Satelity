# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 22:14:14 2021

@author: Maciek
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt 
from matplotlib.pyplot import rc, rcParams, grid 
import matplotlib.patches as mpatches




def plot_skyplot(sat_positions):
    # sat_positions - [PRN, el, az] w stopniach
    rc('grid', color='gray', linewidth=1, linestyle='--')
    fontsize = 20
    rc('xtick', labelsize = fontsize)
    rc('ytick', labelsize = fontsize)
    rc('font', size = fontsize)
    # define colors
    
    green   ='#467821'
    blue    ='#348ABD'
    red     ='#A60628'
    orange  ='#E24A33'
    purple  ='#7A68A6'

        # start ploting
    fig =plt.figure(figsize=(8,6))
    plt.subplots_adjust(bottom= 0.1, 
                        top   = 0.85,
                        left  = 0.1, 
                        right = 0.74)
    ax = fig.add_subplot(polar=True) #define a polar type of coordinates
    ax.set_theta_zero_location('N') # ustawienie kierunku północy na górze wykresu
    ax.set_theta_direction(-1) # ustawienie kierunku przyrostu azymutu w prawo
    
    PG = 0 # zliczanie satelitów GPS
    
    for (PRN, el, az) in sat_positions: 
        PG += 1
        ### show sat number

        ax.annotate(PRN, 
                    xy=(np.radians(az), 90-el),
                    bbox=dict(boxstyle="round", fc = green, alpha = 0.5),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color = 'k')
        #                     # 

    gps     = mpatches.Patch(color=green,  label='{:02.0f}  GPS'.format(PG))
    plt.legend(handles=[gps], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # axis ticks descriptions    
    ax.set_yticks(range(0, 90+10, 10))                   # Define the yticks
    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    ax.set_yticklabels(yLabel)
    # saving and showing plot
    # plt.savefig('satellite_skyplot.pdf')
    plt.show() # wyświetleni
        
    print(PG)










if __name__ == '__main__':
    # sat_positions = np.ones((10,20,3))
    sat_positions = [['PG04', 35, 0], ['PG02', 35, 89], ['PG04', 25, 40],['PG04', 35, 180],['PG04', 35, 130], ['PG24', 35, 270]]
    plot_skyplot(sat_positions)
    
    