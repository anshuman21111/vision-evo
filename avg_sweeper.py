import os
import threading
import subprocess as sub
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
            print(f'files removed for param {p}')

    def rad_simulate(self, param_range=[]):
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
                temp_perc = np.append(temp_perc, np.asarray(self.__trial_loader(p, t)))

            data_perc = pd.DataFrame(temp_perc)
            data_perc.to_csv(f'data/{self.filename}/{self.filename}_{p:2.3f}.csv', index=False)
        
            # Delete old (now summarized) data files
            for t in range(self.trials): 
                os.remove(f'data/{self.filename}/{self.filename}_{p:2.3f}_{t:d}.csv')
            print(f'files removed for param {p}')

    def evaluate(self):
        col_names = ["%2.3f" % p for p in self.param_range]
        data_perc = pd.DataFrame(columns=col_names)
        curr_len = 0

        for i in range(len(self.param_range)):
            agents = self.__eval_loader(self.param_range[i])     # get agents
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
    

    def visualize(self, mode='quantile', ax=None, title_str=None, ymax=None):
        df = pd.read_csv(f'data/{self.filename}/{self.filename}_{self.param}.csv')
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
            ax.set_xlabel(self.param)
            ax.set_ylabel('Perceptual ranges at end time')
            if ymax is not None: ax.set_ylim(top=ymax)
            if title_str is not None: ax.set_title(title_str) 
        else: raise RuntimeError('mode not found, please enter \'quantile\', \'violin\', or \'figure\'')


if __name__ == "__main__":
    deactive = False
    lst = ['energyQuant_rescDen1.5']

    if deactive: #basalEnergyCost and radiusCost in rescDensity
        base075 = Simulation('basal1', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10)
        base025 = Simulation('basal_rescDen0.25', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])
        base15 = Simulation('basal_rescDen1.5', 'basalEnergyCost', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        
        radius075 = Simulation('rad1', 'radiusCost', np.arange(0, 2.1, 0.1), 10)
        radius15 = Simulation('rad2', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '1.5'])
        radius025 = Simulation('rad3', 'radiusCost', np.arange(0, 2.1, 0.1), 10, ARGS=['-rescDensity', '0.25'])

        fig, axes = plt.subplots(nrows=2, ncols=3)
        base025.visualize(mode='figure', ax=axes[0,0], title_str='resourceDensity = 0.25', ymax=2)
        base075.visualize(mode='figure', ax=axes[0,1], title_str='resourceDensity = 0.75', ymax=2)
        base15.visualize(mode='figure', ax=axes[0,2], title_str='resourceDensity = 1.5', ymax=2)

        radius025.visualize(mode='figure', ax=axes[1,0], ymax=2.65)
        radius075.visualize(mode='figure', ax=axes[1,1], ymax=2.65)
        radius15.visualize(mode='figure', ax=axes[1,2], ymax=2.65)

        fig.suptitle('Deactivation parameters')
        fig.set_size_inches(15, 7)
        plt.savefig('deactivate.svg')
        plt.show()
    else: #energyQuantity, growthRate, and maxMutate in rescDensity
        quant025 = Simulation('energyQuant_rescDen0.25', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.25'])
        quant075 = Simulation('energyQuant_rescDen0.75', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '0.75'])
        quant15 = Simulation('energyQuant_rescDen1.5', 'energyQuantity', np.arange(0, 1.5, 0.1), 10, ARGS=['-rescDensity', '1.5'])

        gRate075 = Simulation('grate1', 'growthRate', np.arange(0, 0.8, 0.05), 10)
        gRate15 = Simulation('gRate2', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        gRate025 = Simulation('gRate3', 'growthRate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])

        maxmut075 = Simulation('maxmut1', 'maxMutate', np.arange(0, 0.8, 0.05), 10)
        maxmut15 = Simulation('maxmut2', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '1.5'])
        maxmut025 = Simulation('maxmut3', 'maxMutate', np.arange(0, 0.8, 0.05), 10, ARGS=['-rescDensity', '0.25'])

        fig, axes = plt.subplots(nrows=3, ncols=3)
        quant025.visualize(mode='figure', ax=axes[0,0], title_str='resourceDensity = 0.25', ymax=2.5)
        quant075.visualize(mode='figure', ax=axes[0,1], title_str='resourceDensity = 0.75', ymax=2.5)
        quant15.visualize(mode='figure', ax=axes[0,2], title_str='resourceDensity = 1.5', ymax=2.5)

        gRate025.visualize(mode='figure', ax=axes[1,0], ymax=2.5)
        gRate075.visualize(mode='figure', ax=axes[1,1], ymax=2.5)
        gRate15.visualize(mode='figure', ax=axes[1,2], ymax=2.5)

        maxmut025.visualize(mode='figure', ax=axes[2,0], ymax=3)
        maxmut075.visualize(mode='figure', ax=axes[2,1], ymax=3)
        maxmut15.visualize(mode='figure', ax=axes[2,2], ymax=3)

        fig.suptitle('Activation parameters')
        fig.set_size_inches(15, 10.5)
        plt.savefig('activate.svg')
        plt.show()

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