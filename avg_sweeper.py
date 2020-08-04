import os
import threading
import subprocess as sub
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 12})

class Simulation:
    def __init__(self, filename, param, param_range, trials, ARGS=[]):
        self.filename = filename
        self.param = param
        self.param_range = param_range
        self.trials = trials
        self.ARGS = ARGS
    
    def __trial_loader(self, p, t):
        return pd.read_csv(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv', comment='#')

    def __eval_loader(self, p):
        return pd.read_csv(f'data/{self.filename}/{self.filename}_{p:2.3f}.csv', comment='#')

    def add_param(self, range_to_add, simulate=True, evaluate=True):
        self.param_range += range_to_add
        if simulate: self.simulate(param_range=range_to_add)
        if evaluate: self.exist_evaluate(range_to_add)

    def simulate(self, param_range=[]):
        Path('data/' + self.filename).mkdir(parents=True, exist_ok=True)
        if param_range == []: param_range = self.param_range
        for p in param_range:
            temp_perc = np.empty((1,)) # create temp_perc accumulator array
            for t in range(self.trials): # run simulations (for Windows, change ".out" to ".exe")
                call_str = ['./capped_energy_resc_refill.out', f'-{self.param}', f'{p:2.3f}'] + self.ARGS
                outfile = open(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv', 'w+')
                sub.run(call_str, stdout=outfile)
                outfile.close()

                # Evaluate all the trials into one file for the param value
                df = self.__trial_loader(p, t)
                agents = df[df['Type'] == 0]       # get agents
                endtime = max(agents['Iteration']) # get end time

                # Extract perceptual ranges at end time and add to accumulator
                endtime_rows = agents[agents['Iteration'] == endtime]
                temp_perc = np.append(temp_perc, np.asarray(endtime_rows['Radius']))

            data_perc = pd.DataFrame(temp_perc)
            data_perc.to_csv(f'data/{self.filename}/{self.filename}_{p:2.3f}.csv', index=False)
        
            # Delete old (now summarized) data files
            for t in range(self.trials): 
                os.remove(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv')

    def rad_simulate(self, param_range=[]):
        Path('data/' + self.filename).mkdir(parents=True, exist_ok=True)
        if param_range == []: param_range = self.param_range
        for p in tqdm(param_range):
            temp_perc = {}   # create temp_perc accumulator
            for t in range(self.trials): # run simulations (for Windows, change ".out" to ".exe")
                call_str = ['./capped_energy_resc_refill.out', f'-{self.param}', f'{p:2.3f}'] + self.ARGS
                outfile = open(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv', 'w+')
                sub.run(call_str, stdout=outfile)
                outfile.close()

                # Evaluate all the trials into one file for the param value
                tr = self.__trial_loader(p, t)
                if tr.empty: 
                    temp_perc[f'{t:d}'] = np.array([0])
                else:
                    temp_perc[f'{t:d}'] = tr.values.flatten()

            # Resize all lists
            max_len = max([len(lst) for lst in temp_perc.values()])
            for k, lst in temp_perc.items():
                if max_len != len(lst): temp_perc[k] = np.append(lst, [np.nan]*(max_len - len(lst)))

            data_perc = pd.DataFrame(temp_perc)
            data_perc.to_csv(f'data/{self.filename}/{self.filename}_{p:2.3f}.csv', index=False)
        
            # Delete old (now summarized) data files
            for t in range(self.trials): 
                os.remove(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv')

    def evaluate(self):
        col_names = ["%2.3f" % p for p in self.param_range]
        temp_perc = {}

        for i in range(len(self.param_range)):
            agents = self.__eval_loader(self.param_range[i])     # get agents
            temp_perc[i] = agents[agents.notna()].values.flatten()

        # Resize all lists
        max_len = max([len(lst) for lst in temp_perc.values()])
        for k, lst in temp_perc.items():
            if max_len != len(lst): temp_perc[k] = np.append(lst, [np.nan]*(max_len - len(lst)))

        data_perc = pd.DataFrame(temp_perc)
        data_perc.to_csv(f'data/{self.filename}/{self.filename}_{self.param}.csv', index=False)
    
        # Delete old (now summarized) data files
        for p in col_names: os.remove(f'data/{self.filename}/{self.filename}_{p}.csv')
    
    def exist_evaluate(self, range_to_add):
        filenames = ["%2.3f" % p for p in range_to_add]
        data_perc = pd.read_csv(f'data/{self.filename}/{self.filename}_{self.param}.csv')
        curr_len = 0

        for i in range(len(range_to_add)):
            agents = self.__eval_loader(range_to_add[i])     # get agents
            temp_perc = np.asarray(agents)

            # Adjust static matrix sizes
            if len(temp_perc) > curr_len: 
                curr_len = len(temp_perc)
                data_perc = pd.DataFrame(np.resize(data_perc.values, (curr_len, len(self.param_range))))
            elif len(temp_perc) < curr_len:
                temp_perc = np.append(temp_perc, np.full((curr_len - len(temp_perc),1), np.nan), axis=0)

            data_perc[i] = temp_perc

        data_perc.to_csv(f'data/{self.filename}/{self.filename}_{self.param}.csv', index=False)
        
        # Delete old (now summarized) data files
        for p in filenames: os.remove(f'data/{self.filename}/{self.filename}_{p}.csv')
    
    def get_thres(self):
        # Returns (threshold value, True/False) where the boolean 
        # corresponds to whether or not this is an activation threshold.
        # If there is no threshold, return None.

        df = pd.read_csv(f'./data/{self.filename}/{self.filename}_{self.param}.csv')
        qs = df.quantile([0.025, 0.25, 0.5, 0.75, 0.975]).values
        
        # Find where ranges get activated or deactivated
        eps = np.finfo(float).eps #machine epsilon
        for i in range(qs.shape[1] - 1):
            if sum(qs[:,i]) < eps and sum(qs[:,i+1]) > eps:
                return (self.param_range[i], True)
            elif sum(qs[:,i]) > eps and sum(qs[:,i+1]) < eps:
                return (self.param_range[i+1], False)
        return None
    
    def summarize(self):
        # Read data file and get quantiles
        df = pd.read_csv(f'./data/{self.filename}/{self.filename}_{self.param}.csv')
        intervals = [0.025, 0.25, 0.5, 0.75, 0.975]
        qs = df.quantile(intervals)
        
        # Rearrange data (response variables)
        data = qs.transpose()

        # Add explanatory variable columns
        repi = None
        radi = None
        resi = None
        grai = None
        maxi = None
        enei = None
        basi = None
        N = len(self.param_range)
        o = np.ones((N,))

        for i in range(len(self.ARGS)):
            s = self.ARGS[i]
            if 'reproCost' in s:       repi = i
            if 'radiusCost' in s:      radi = i
            if 'basalEnergyCost' in s: basi = i
            if 'rescDensity' in s:     resi = i
            if 'growthRate' in s:      grai = i
            if 'maxMutate' in s:       maxi = i
            if 'energyQuantity' in s:  enei = i

        data.insert(0, f'{self.param}', self.param_range)

        if self.param != 'rescDensity':
            if resi is not None:
                data.insert(0, 'rescDensity', float(self.ARGS[resi+1])*o)
            else:
                data.insert(0, 'rescDensity', 0.75*o)

        if self.param != 'growthRate':
            if grai is not None:
                data.insert(0, 'growthRate', float(self.ARGS[grai+1])*o)
            else:
                data.insert(0, 'growthRate', 0.5*o)

        if self.param != 'maxMutate':
            if maxi is not None:
                data.insert(0, 'maxMutate', float(self.ARGS[maxi+1])*o)
            else:
                data.insert(0, 'maxMutate', 0.25*o)
        
        if self.param != 'energyQuantity':
            if enei is not None:
                data.insert(0, 'energyQuantity', float(self.ARGS[enei+1])*o)
            else:
                data.insert(0, 'energyQuantity', o)
        
        if self.param != 'basalEnergyCost':
            if basi is not None:
                data.insert(0, 'basalEnergyCost', float(self.ARGS[basi+1])*o)
            else:
                data.insert(0, 'basalEnergyCost', 0.05*o)
        
        if self.param != 'radiusCost':
            if radi is not None:
                data.insert(0, 'radiusCost', float(self.ARGS[radi+1])*o)
            else:
                data.insert(0, 'radiusCost', 0.1*o)
        
        if self.param != 'reproCost':
            if repi is not None:
                data.insert(0, 'reproCost', float(self.ARGS[repi+1])*o)
            else:
                data.insert(0, 'reproCost', 0.5*o)

        data.to_csv(f'./data/{self.filename}/summary.csv', index=False)

    def visualize(self, mode='quantile', ax=None, title_str=None, rem_ticks=False, ymax=None):
        df = pd.read_csv(f'./data/{self.filename}/{self.filename}_{self.param}.csv')
        if mode == 'quantile':
            intervals = [0.025, 0.25, 0.5, 0.75, 0.975]
            qs = df.quantile(intervals)
            
            plt.plot(self.param_range, qs.loc[0.025], color='blue')
            plt.plot(self.param_range, qs.loc[0.975], color='blue')
            plt.plot(self.param_range, qs.loc[0.25], color='orange')
            plt.plot(self.param_range, qs.loc[0.75], color='orange')

            plt.fill_between(self.param_range, qs.loc[0.025], qs.loc[0.25], alpha=0.15)
            plt.fill_between(self.param_range, qs.loc[0.25], qs.loc[0.75], alpha=0.35)
            plt.fill_between(self.param_range, qs.loc[0.75], qs.loc[0.975], alpha=0.15)

            plt.plot(self.param_range, qs.loc[0.5], color='black')
            plt.xlabel(self.param)
            plt.ylabel('Perceptual ranges at end time')
            plt.title(f'Effect of {self.param} on perceptual ranges over time')
            plt.show()       
        elif mode == 'violin':
            sns.violinplot(data=df, bw=.1, inner=None, cut=0, scale='width')
            plt.show()
        elif mode == 'figure':
            intervals = [0.025, 0.25, 0.5, 0.75, 0.975]
            qs = df.quantile(intervals)
            
            ax.plot(self.param_range, qs.loc[0.025], color='blue')
            ax.plot(self.param_range, qs.loc[0.975], color='blue')
            ax.plot(self.param_range, qs.loc[0.25], color='orange')
            ax.plot(self.param_range, qs.loc[0.75], color='orange')

            ax.fill_between(self.param_range, qs.loc[0.025], qs.loc[0.25], alpha=0.15)
            ax.fill_between(self.param_range, qs.loc[0.25], qs.loc[0.75], alpha=0.35)
            ax.fill_between(self.param_range, qs.loc[0.75], qs.loc[0.975], alpha=0.15)

            ax.plot(self.param_range, qs.loc[0.5], color='black')
            ax.tick_params(axis='x', labelrotation=30)
            if rem_ticks: ax.set_yticks([])
            if ymax is not None: ax.set_ylim(top=ymax)
            if title_str is not None: ax.set_title(title_str) 
        else: raise RuntimeError('mode not found, please enter \'quantile\', \'violin\', or \'figure\'')

def make_heatmap(filebase, xparam, param_range, yparam='rescDensity', ax=None):
    if yparam == 'rescDensity': 
        fileparam = 'rescDen'
        yticks = ['0.25', '0.75', '1.5']
    elif yparam == 'growthRate': 
        fileparam = 'gRate'
        yticks = ['0.05', '0.5', '0.75']
    else: raise ValueError('mode not found, enter \'rescDensity\' or \'growthRate\'')
    
    filenames = [filebase + '_' + fileparam + num for num in ['0.25', '0.75', '1.5']]
    low = pd.read_csv(f'./data/{filenames[0]}/{filenames[0]}_{xparam}.csv')
    med = pd.read_csv(f'./data/{filenames[1]}/{filenames[1]}_{xparam}.csv')
    hi  = pd.read_csv(f'./data/{filenames[2]}/{filenames[2]}_{xparam}.csv')
    dfs = [low, med, hi]

    # Find where ranges are activated or deactivated
    data = np.zeros((3, param_range.shape[0]))
    for j in range(3):
        data[j,:] = dfs[j].quantile([0.5]).values

    show = False
    if ax is None: 
        ax = plt.axes()
        show = True

    im = ax.imshow(data, origin='lower')
    ax.set_xlabel(xparam); ax.set_ylabel(yparam)
    ax.set_xticks(range(len(param_range)))
    ax.set_xticklabels(['%1.2f' % num for num in param_range]); 
    ax.set_yticks(range(3)); ax.set_yticklabels(yticks)
    
    # Set colorbar size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if show: plt.show()

if __name__ == "__main__":
    active = False
    deactive = False
    hmap = True
    make_csv = False

    if deactive: #basalEnergyCost and radiusCost in rescDensity
        base075 = Simulation('basal_grate0.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10)
        base025 = Simulation('basal_rescDen0.25', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        base15 = Simulation('basal_rescDen1.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        
        radius075 = Simulation('rad_rescDen0.75', 'radiusCost', np.arange(0, 2.1, 0.1), 10)
        radius15 = Simulation('rad_rescDen1.5', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '1.5'])
        radius025 = Simulation('rad_rescDen0.25', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '0.25'])

        fig, axes = plt.subplots(nrows=2, ncols=3)
        ax = fig.add_subplot(111, frameon=False) #big overall subplot for labels
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_ylabel('Perceptual ranges at end time', fontsize=18)

        bax = fig.add_subplot(211, frameon=False)
        bax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bax.set_xlabel('basalEnergyCost', fontsize=18)

        rax = fig.add_subplot(212, frameon=False)
        rax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        rax.set_xlabel('radiusCost', fontsize=18)

        base025.visualize(mode='figure', ax=axes[0,0], ymax=2)
        base075.visualize(mode='figure', ax=axes[0,1], ymax=2)
        base15.visualize(mode='figure', ax=axes[0,2], ymax=2)

        radius025.visualize(mode='figure', ax=axes[1,0], ymax=2.65)
        radius075.visualize(mode='figure', ax=axes[1,1], ymax=2.65)
        radius15.visualize(mode='figure', ax=axes[1,2], ymax=2.65)

        fig.set_size_inches(15, 7)
        plt.subplots_adjust(hspace=0.45, wspace=0.15)
        plt.savefig('deactivate.png')
        plt.show()
    elif active: #energyQuantity, growthRate, and maxMutate in rescDensity
        quant025 = Simulation('energyQuant_rescDen0.25', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
        quant075 = Simulation('energyQuant_rescDen0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
        quant15 = Simulation('energyQuant_rescDen1.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])

        gRate075 = Simulation('grate_rescDen0.75', 'growthRate', np.arange(0, 0.8, 0.05), 10)
        gRate15 = Simulation('gRate_rescDen1.5', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        gRate025 = Simulation('gRate_rescDen0.25', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])

        maxmut075 = Simulation('maxmut_rescDen0.75', 'maxMutate', np.arange(0, 0.8, 0.05), 10)
        maxmut15 = Simulation('maxmut_rescDen1.5', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        maxmut025 = Simulation('maxmut_rescDen0.25', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])

        fig, axes = plt.subplots(nrows=3, ncols=3)
        ax = fig.add_subplot(111, frameon=False) #big overall subplot for labels
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel('Perceptual ranges at end time', labelpad=50, fontsize=18)

        eax = fig.add_subplot(311, frameon=False)
        eax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        eax.set_xlabel('energyQuantity', labelpad=10, fontsize=18)

        gax = fig.add_subplot(312, frameon=False)
        gax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        gax.set_xlabel('growthRate', labelpad=10, fontsize=18)

        mmx = fig.add_subplot(313, frameon=False)
        mmx.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        mmx.set_xlabel('maxMutate', labelpad=10, fontsize=18)

        quant025.visualize(mode='figure', ax=axes[0,0], ymax=2.5)
        quant075.visualize(mode='figure', ax=axes[0,1], ymax=2.5, rem_ticks=True)
        quant15.visualize(mode='figure', ax=axes[0,2], ymax=2.5, rem_ticks=True)

        gRate025.visualize(mode='figure', ax=axes[1,0], ymax=2.5)
        gRate075.visualize(mode='figure', ax=axes[1,1], ymax=2.5, rem_ticks=True)
        gRate15.visualize(mode='figure', ax=axes[1,2], ymax=2.5, rem_ticks=True)

        maxmut025.visualize(mode='figure', ax=axes[2,0], ymax=3)
        maxmut075.visualize(mode='figure', ax=axes[2,1], ymax=3, rem_ticks=True)
        maxmut15.visualize(mode='figure', ax=axes[2,2], ymax=3, rem_ticks=True)

        fig.set_size_inches(15, 10.5)
        plt.subplots_adjust(hspace=0.45, wspace=0.15)
        plt.savefig('activate.png')
        plt.show()
    elif hmap: 
        _, (ax1, ax2) = plt.subplots(figsize=(9.6,4.5), nrows=2)#, gridspec_kw={'height_ratios': [1, 1]})
        make_heatmap('basal', 'basalEnergyCost', np.arange(0, 0.8, 0.05), ax=ax1)
        make_heatmap('rad', 'radiusCost', np.arange(0, 2.1, 0.1), ax=ax2)
        plt.suptitle('Mean distribution values, deactivation parameters')
        plt.savefig('../pics/paper_plots/deactive_means.pdf')
        plt.show()

        _, (ax1, ax2, ax3) = plt.subplots(figsize=(9,6.5), nrows=3)#, gridspec_kw={'height_ratios': [1, 1, 1]})
        make_heatmap('energyQuant', 'energyQuantity', np.arange(0, 1.5, 0.1), ax=ax1)
        make_heatmap('grate', 'growthRate', np.arange(0, 0.8, 0.05), ax=ax2)
        make_heatmap('maxmut', 'maxMutate', np.arange(0, 0.8, 0.05), ax=ax3)
        plt.suptitle('Mean distribution values, activation parameters')
        plt.savefig('../pics/paper_plots/active_means.pdf')
        plt.show()
    elif make_csv:
        base = Simulation('basal_rescDen0.25', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        base.summarize()

        base = Simulation('basal_rescDen1.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        base.summarize()

        base = Simulation('basal_grate0.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.5'])
        base.summarize()
        
        base = Simulation('basal_grate0.75', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
        base.summarize()

        base = Simulation('basal_grate0.05', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.05'])
        base.summarize()

        base = Simulation('basal_grate0.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.5'])
        base.summarize()
        
        base = Simulation('basal_grate0.75', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
        base.summarize()

        gather = Simulation('energyQuant_grate0.05', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.05'])
        gather.summarize()

        gather = Simulation('energyQuant_grate0.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.5'])
        gather.summarize()
        
        gather = Simulation('energyQuant_grate0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.75'])
        gather.summarize()

        gather = Simulation('energyQuant_rescDen0.25', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
        gather.summarize()

        gather = Simulation('energyQuant_rescDen0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
        gather.summarize()
        
        gather = Simulation('energyQuant_rescDen1.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])
        gather.summarize()
        
        gRate = Simulation('grate_rescDen0.75', 'growthRate', np.arange(0, 0.8, 0.05), 10)
        gRate.summarize()
        
        gRate = Simulation('gRate_rescDen0.25', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        gRate.summarize()
        
        gRate = Simulation('gRate_rescDen1.5', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        gRate.summarize()

        maxmut075 = Simulation('maxmut_grate0.05', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.05'])
        maxmut075.summarize()

        maxmut075 = Simulation('maxmut_grate0.5', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.5'])
        maxmut075.summarize()

        maxmut075 = Simulation('maxmut_grate0.75', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
        maxmut075.summarize()
        
        maxmut15 = Simulation('maxmut_rescDen1.5', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        maxmut15.summarize()

        maxmut025 = Simulation('maxmut_rescDen0.25', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        maxmut025.summarize()
        
        maxmut075 = Simulation('maxmut_rescDen0.75', 'maxMutate', np.arange(0, 0.8, 0.05), 10)
        maxmut075.summarize()

        rad = Simulation('maxmut_rescDen0.25', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        rad.summarize()

        radius = Simulation('rad_grate0.75', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.75'])
        radius.summarize()

        radius = Simulation('rad_grate0.05', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.05'])
        radius.summarize()

        radius = Simulation('rad_grate0.5', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.5'])
        radius.summarize()

        radius = Simulation('rad_rescDen0.75', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '0.75'])
        radius.summarize()

        radius = Simulation('rad_rescDen0.25', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '0.25'])
        radius.summarize()

        radius = Simulation('rad_rescDen1.5', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '1.5'])
        radius.summarize()

        repro = Simulation('repro_rescDen0.25', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        repro.summarize()

        repro = Simulation('repro_rescDen0.75', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.75'])
        repro.summarize()

        repro = Simulation('repro_rescDen1.5', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        repro.summarize()

        repro = Simulation('repro_grate0.05', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.05'])
        repro.summarize()

        repro = Simulation('repro_grate0.5', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.5'])
        repro.summarize()

        repro = Simulation('repro_grate0.75', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.75'])
        repro.summarize()

        stubs = ['basal_rescDen0.25', 'basal_rescDen1.5', 'basal_grate0.5', 'basal_grate0.75', \
            'basal_grate0.05', 'basal_grate0.5', 'basal_grate0.75', \
            'energyQuant_rescDen0.25', 'energyQuant_rescDen0.75', 'energyQuant_rescDen1.5', \
            'gRate_rescDen0.25', 'grate_rescDen0.75', 'gRate_rescDen1.5', 'maxmut_rescDen0.25', \
            'maxmut_rescDen0.75', 'maxmut_rescDen1.5', 'maxmut_grate0.05', 'maxmut_grate0.5', \
            'maxmut_grate0.75', 'rad_grate0.05', 'rad_grate0.75', 'rad_grate0.5', 'rad_rescDen0.75', \
            'rad_rescDen0.25', 'rad_rescDen1.5', 'repro_rescDen0.25', 'repro_rescDen0.75', \
            'repro_grate0.05', 'repro_grate0.5', 'repro_grate0.75']
        filenames = ['./data/' + stub + '/summary.csv' for stub in stubs]
        combined_df = pd.concat([pd.read_csv(f) for f in filenames])
        combined_df.to_csv('./data/activation_summary.csv')
    else:
        lst = ['gather', 'grate', 'maxmut']
        for selector in lst:
            if selector == 'gather':
                gather = Simulation('energyQuant_rescDen0.25', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
                gather.visualize()

                gather = Simulation('energyQuant_rescDen0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
                gather.visualize()

                gather = Simulation('energyQuant_rescDen1.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])
                gather.visualize()
            elif selector == 'grate':
                gRate = Simulation('grate_rescDen0.75', 'growthRate', np.arange(0, 0.8, 0.05), 10)
                gRate.visualize()

                gRate = Simulation('gRate_rescDen0.25', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
                gRate.visualize()

                gRate = Simulation('gRate_rescDen1.5', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
                gRate.visualize()
            elif selector == 'maxmut':
                maxmut075 = Simulation('maxmut_rescDen0.75', 'maxMutate', np.arange(0, 0.8, 0.05), 10)
                maxmut075.visualize()

                maxmut15 = Simulation('maxmut_rescDen1.5', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
                maxmut15.visualize()

                maxmut025 = Simulation('maxmut_rescDen0.25', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
                maxmut025.visualize()

    # Uncomment the following to run in bulk
    """ for selector in lst:
        if selector == 'grate1':
            gRate = Simulation('grate1', 'growthRate', np.arange(0, 0.8, 0.05), 10)
            gRate.visualize()
        elif selector == 'grate2':
            gRate = Simulation('gRate2', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
            gRate.visualize()
        elif selector == 'grate3':
            gRate = Simulation('gRate3', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
            gRate.visualize()
        elif selector == 'maxmut1':
            maxmut = Simulation('maxmut1', 'maxMutate', np.arange(0, 0.8, 0.05), 10)
            maxmut.visualize()
        elif selector == 'maxmut2':
            maxmut = Simulation('maxmut2', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
            maxmut.visualize()
        elif selector == 'maxmut3':
            maxmut = Simulation('maxmut3', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
            maxmut.visualize()
        elif selector == 'maxmut_lowgrate':
            maxmut = Simulation('maxmut_lowgrate', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.1'])
            maxmut.visualize()
        elif selector == 'maxmut_higrate':
            maxmut = Simulation('maxmut_higrate', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
            maxmut.visualize()
        elif selector == 'resc1':
            resc = Simulation('resc1', 'rescDensity', np.arange(0, 2.1, 0.1), 10)
            resc.visualize()
        elif selector == 'rad1':
            radius = Simulation('rad1', 'radiusCost', np.arange(0, 2.1, 0.1), 10)
            radius.visualize()
        elif selector == 'rad2':
            radius = Simulation('rad2', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '1.5'])
            radius.visualize()
        elif selector == 'rad3':
            radius = Simulation('rad3', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '0.25'])
            radius.visualize()
        elif selector == 'rad_grate0.25':
            radius = Simulation('rad_grate0.25', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.25'])
            radius.visualize()
        elif selector == 'rad_grate0.5':
            radius = Simulation('rad_grate0.5', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.5'])
            radius.visualize()
        elif selector == 'rad_grate0.75':
            radius = Simulation('rad_grate0.75', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-growthRate', '0.75'])
            radius.visualize()
        elif selector == 'distTest':
            distTest = Simulation('distTest', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-gatherDist', '0.1'])
            distTest.visualize()
        elif selector == 'distTest0.5':
            distTest = Simulation('distTest0.5', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-gatherDist', '0.5'])
            distTest.visualize()
        elif selector == 'distTest_rg.1':
            distTest = Simulation('distTest_rg.1', 'rescDensity', np.arange(0, 1.5, 0.1), 10, ARGS=['-gatherDist', '0.1'])
            distTest.visualize()
        elif selector == 'distTest_rg.5':
            distTest = Simulation('distTest_rg.5', 'rescDensity', np.arange(0, 1.5, 0.1), 10, ARGS=['-gatherDist', '0.5'])
            distTest.visualize()
        elif selector == 'basal_rescDen0.75':
            baseTest = Simulation('basal1', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10)
            baseTest.visualize()
        elif selector == 'basal_rescDen0.25':
            baseTest = Simulation('basal_rescDen0.25', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
            baseTest.visualize()
        elif selector == 'basal_rescDen1.5':
            baseTest = Simulation('basal_rescDen1.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
            baseTest.visualize()
        elif selector == 'reproCost_grate0.5':
            reproTest = Simulation('repro_grate0.5', 'reproCost', np.arange(0, 0.8, 0.05), 10)
            reproTest.visualize()
        elif selector == 'reproCost_grate0.05':
            reproTest = Simulation('repro_grate0.05', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.05'])
            reproTest.visualize()
        elif selector == 'reproCost_grate0.75':
            reproTest = Simulation('repro_grate0.75', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
            reproTest.visualize()
        elif selector == 'gatherDist_grate0.05':
            gather = Simulation('gatherDist_grate0.05', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.05'])
            gather.visualize()
        elif selector == 'gatherDist_grate0.5':
            gather = Simulation('gatherDist_grate0.5', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.5'])
            gather.visualize()
        elif selector == 'gatherDist_grate0.75':
            gather = Simulation('gatherDist_grate0.75', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
            gather.visualize()
        elif selector == 'gatherAmt_grate0.05':
            gatherA = Simulation('gatherAmt_grate0.05', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.05'])
            gatherA.visualize()
        elif selector == 'gatherAmt_grate0.5':
            gatherA = Simulation('gatherAmt_grate0.5', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.5'])
            gatherA.visualize() 
        elif selector == 'gatherAmt_grate0.75':
            gatherA = Simulation('gatherAmt_grate0.75', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.75'])
            gatherA.visualize()
        elif selector == 'energyCap_grate0.05':
            cap = Simulation('energyCap_grate0.05', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-growthRate', '0.05'])
            cap.visualize()
        elif selector == 'energyCap_grate0.5':
            cap = Simulation('energyCap_grate0.5', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-growthRate', '0.5'])
            cap.visualize()
        elif selector == 'energyCap_grate0.75':
            cap = Simulation('energyCap_grate0.75', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-growthRate', '0.75'])
            cap.visualize()
        elif selector == 'basal_grate0.05':
            baseTest = Simulation('basal_grate0.05', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.05'])
            baseTest.visualize()
        elif selector == 'basal_grate0.25':
            baseTest = Simulation('basal_grate0.25', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.25'])
            baseTest.visualize()
        elif selector == 'basal_grate0.5':
            baseTest = Simulation('basal_grate0.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.5'])
            baseTest.visualize()
        elif selector == 'basal_grate0.75':
            baseTest = Simulation('basal_grate0.75', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-growthRate', '0.75'])
            baseTest.visualize()
        elif selector == 'gatherDist_rescDen0.25':
            gather = Simulation('gatherDist_rescDen0.25', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
            gather.visualize()
        elif selector == 'gatherDist_rescDen0.75':
            gather = Simulation('gatherDist_rescDen0.75', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.75'])
            gather.visualize()
        elif selector == 'gatherDist_rescDen1.5':
            gatherA = Simulation('gatherDist_rescDen1.5', 'gatherDist', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
            gatherA.visualize()
        elif selector == 'gatherAmt_rescDen0.25':
            gather = Simulation('gatherAmt_rescDen0.25', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
            gather.visualize()
        elif selector == 'gatherAmt_rescDen0.75':
            gather = Simulation('gatherAmt_rescDen0.75', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
            gather.visualize()
        elif selector == 'gatherAmt_rescDen1.5':
            gatherA = Simulation('gatherAmt_rescDen1.5', 'gatherAmount', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])
            gatherA.visualize()
        elif selector == 'energyQuant_grate0.5':
            gather = Simulation('energyQuant_grate0.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.5'])
            gather.visualize()
        elif selector == 'energyQuant_grate0.75':
            gather = Simulation('energyQuant_grate0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.75'])
            gather.visualize()
        elif selector == 'energyQuant_grate0.05':
            gatherA = Simulation('energyQuant_grate0.05', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-growthRate', '0.05'])
            gatherA.visualize()
        elif selector == 'energyQuant_rescDen0.25':
            gather = Simulation('energyQuant_rescDen0.25', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
            gather.visualize()
        elif selector == 'energyQuant_rescDen0.75':
            gather = Simulation('energyQuant_rescDen0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
            gather.visualize()
        elif selector == 'energyQuant_rescDen1.5':
            gatherA = Simulation('energyQuant_rescDen1.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])
            gatherA.visualize()
        elif selector == 'energyCap_rescDen0.25':
            cap = Simulation('energyCap_rescDen0.25', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-rescDensity', '0.25'])
            cap.visualize()
        elif selector == 'energyCap_rescDen0.75':
            cap = Simulation('energyCap_rescDen0.75', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-rescDensity', '0.75'])
            cap.visualize()
        elif selector == 'energyCap_rescDen1.5':
            cap = Simulation('energyCap_rescDen1.5', 'energyCap', np.arange(0.5, 5.25, 0.25), 10, ARGS=['-rescDensity', '1.5'])
            cap.visualize()
        elif selector == 'reproCost_rescDen0.25':
            reproTest = Simulation('repro_rescDen0.25', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
            reproTest.visualize()
        elif selector == 'reproCost_rescDen0.75':
            reproTest = Simulation('repro_rescDen0.75', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.75'])
            reproTest.visualize()
        elif selector == 'reproCost_rescDen1.5':
            reproTest = Simulation('repro_rescDen1.5', 'reproCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
            reproTest.visualize() """