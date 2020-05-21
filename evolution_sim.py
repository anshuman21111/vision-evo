import os
import subprocess as sub
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, ranges, percents, filename, trials, ARGS=None):
        self.ranges = ranges
        self.percents = percents
        self.filename = filename
        self.trials = trials
        self.ARGS = ARGS if ARGS is not None else []
    
    def simulate(self): big_simulate(self.ranges, self.percents, self.filename, self.trials, self.ARGS)
    def evaluate(self): big_evaluate(self.ranges, self.percents, self.filename)
    def visualize(self): big_visualize(self.ranges, self.percents, self.filename)

    def sep_visualize(self, ax):
        for r in self.ranges:
            df = evaluated_loader(self.filename, "%2.3f" % r)

            # Count zero ranges and nonzero ranges at each percent
            nonzero_ranges = df.mask(df == 0).count()
            zero_ranges = df.mask(df > 0).count()
            nonzero_prop = nonzero_ranges / (nonzero_ranges + zero_ranges)
            zero_prop = zero_ranges / (nonzero_ranges + zero_ranges)         

            ax.plot(self.percents, nonzero_prop * 100)
            ax.plot(self.percents, zero_prop * 100)
            ax.legend(['nonzero ranges', 'zero ranges'])
            
            ax.set_xlabel('percentage of foragers starting with range ' + "%2.3f" % r)
            ax.set_ylabel('percentage of foragers at end time')
            ax.set_title('Fixed range ' + "%2.3f" % r)

def evaluated_loader(filename, p):
    return pd.read_csv(f'data/{filename}/{filename}{p}.csv', comment='#')

def loader(filename, p, num):
    return pd.read_csv(f'data/{filename}/{filename}{p}_{num}.csv', comment='#')

def trial_loader(filename, p, num, t):
    return pd.read_csv(f'data/{filename}/{filename}{p}_{num}_{t}.csv', comment='#')

def simulate(ranges, percents, filename):    
    Path('data/' + filename).mkdir(parents=True, exist_ok=True)
    popsize = "%d" % 100

    for ran in ranges:
        str_ran = "%2.3f" % ran
        for per in percents: # run simulations (for Windows, change ".out" to ".exe")
            str_per = "%2.3f" % (per/100) 
            call_str = ['./evolution_sim.out', '-fixedRange', str_ran, '-percDec', str_per]
            outfile = open(f'data/{filename}/{filename}{str_ran}_{str_per}.csv', 'w+')
            sub.run(call_str, shell=True, stdout=outfile)
            outfile.close()

def evaluate(ranges, percents, filename):
    col_names = [str(i) for i in percents]
    data_perc = pd.DataFrame(columns=col_names)

    for ran in ranges:
        str_ran = "%2.3f" % ran
        curr_len = 0
        for i in range(len(percents)):
            str_per = "%2.3f" % (percents[i]/100)
            agents = loader(filename, str_ran, str_per)
            #agents = df[df['Type'] == 0]                    # get agents
            sattime = max(agents.mode()['Iteration'][0], 0) # get saturated time

            # Extract perceptual ranges at saturated time
            sattime_rows = agents[agents['Iteration'] == sattime]
            temp_perc = np.asarray(sattime_rows['Radius'])

            # Adjust static matrix sizes
            if len(temp_perc) > curr_len: 
                curr_len = len(temp_perc)
                data_perc = pd.DataFrame(np.resize(data_perc.values, (curr_len, len(percents))))
            elif len(temp_perc) < curr_len:
                temp_perc = np.append(temp_perc, np.full((curr_len - len(temp_perc),), np.nan), axis=0)

            data_perc[i] = temp_perc

        data_perc.to_csv(f'data/{filename}/{filename}{str_ran}.csv', index=False)
    
    # Delete old (now summarized) data files
    for ran in ["%2.3f" % num for num in ranges]:
        for per in ["%2.3f" % (num/100) for num in percents]:
            os.remove(f'data/{filename}/{filename}{ran}_{per}.csv')

def visualize(ranges, percents, filename):
    for r in ranges:
        df = evaluated_loader(filename, "%2.3f" % r)

        # Count zero ranges and nonzero ranges at each percent
        nonzero_df = df.mask(df == 0)
        zero_df = df.mask(df > 0)
        nonzero_ranges = nonzero_df.count()
        zero_ranges = zero_df.count()

        plt.plot(percents, nonzero_ranges)
        plt.plot(percents, zero_ranges)
        plt.legend(['nonzero ranges', 'zero ranges'])
        
        plt.xlabel('percentage of foragers starting with range ' + "%2.3f" % r)
        plt.ylabel('number of foragers')
        plt.title('Fixed range ' + "%2.3f" % r)
        plt.show()

def big_simulate(ranges, percents, filename, trials, ARGS):    
    Path('data/' + filename).mkdir(parents=True, exist_ok=True)

    for ran in ["%2.3f" % num for num in ranges]:
        for per in ["%2.3f" % (num/100) for num in percents]:
            for t in range(trials):
                str_t = "%d" % t
                call_str = ['./evolution_sim.out', '-fixedRange', ran, '-percDec', per] + ARGS
                outfile = open(f'data/{filename}/{filename}{ran}_{per}_{str_t}.csv', 'w+')
                sub.run(call_str, stdout=outfile)
                print('finished')
                outfile.close()
            mid_evaluate(ran, per, filename, trials)
            print(f'simulation range={ran}, percent={per} completed')

def big_evaluate(ranges, percents, filename):
    col_names = [str(i) for i in percents]
    data_perc = pd.DataFrame(columns=col_names)

    for ran in ranges:
        str_ran = "%2.3f" % ran
        curr_len = 0
        for i in range(len(percents)):
            str_per = "%2.3f" % (percents[i]/100)
            agents = loader(filename, str_ran, str_per)     # get agents
            temp_perc = np.asarray(agents)

            # Adjust static matrix sizes
            if len(temp_perc) > curr_len: 
                curr_len = len(temp_perc)
                data_perc = pd.DataFrame(np.resize(data_perc.values, (curr_len, len(percents))))
            elif len(temp_perc) < curr_len:
                temp_perc = np.append(temp_perc, np.full((curr_len - len(temp_perc),1), np.nan), axis=0)

            data_perc[i] = temp_perc

        data_perc.to_csv(f'data/{filename}/{filename}{str_ran}.csv', index=False)
    
    # Delete old (now summarized) data files
    for ran in ["%2.3f" % num for num in ranges]:
        for per in ["%2.3f" % (num/100) for num in percents]:
            os.remove(f'data/{filename}/{filename}{ran}_{per}.csv')

def mid_evaluate(str_ran, str_per, filename, trials):
    temp_perc = np.empty((1,)) # create temp_perc accumulator array
    for t in range(trials):
        str_t = "%d" % t
        df = trial_loader(filename, str_ran, str_per, str_t)
        agents = df[df['Type'] == 0]                    # get agents
        sattime = max(agents.mode()['Iteration'][0], 1) # get saturated time

        # Extract perceptual ranges at saturated time and add to accumulator
        sattime_rows = agents[agents['Iteration'] == sattime]
        temp_perc = np.append(temp_perc, np.asarray(sattime_rows['Radius']))

    data_perc = pd.DataFrame(temp_perc)
    data_perc.to_csv(f'data/{filename}/{filename}{str_ran}_{str_per}.csv', index=False)
    
    # Delete old (now summarized) data files
    for t in [str(num) for num in range(trials)]: 
        os.remove(f'data/{filename}/{filename}{str_ran}_{str_per}_{t}.csv')

def big_visualize(ranges, percents, filename):
    for r in ranges:
        df = evaluated_loader(filename, "%2.3f" % r)

        # Count zero ranges and nonzero ranges at each percent
        nonzero_ranges = df.mask(df == 0).count()
        zero_ranges = df.mask(df > 0).count()
        nonzero_prop = nonzero_ranges / (nonzero_ranges + zero_ranges)
        zero_prop = zero_ranges / (nonzero_ranges + zero_ranges)         

        plt.plot(percents, nonzero_prop * 100)
        plt.plot(percents, zero_prop * 100)
        plt.legend(['nonzero ranges', 'zero ranges'])
        
        plt.xlabel('percentage of foragers starting with range ' + "%2.3f" % r)
        plt.ylabel('percentage of foragers at end time')
        plt.title('Fixed range ' + "%2.3f" % r)
        plt.show()

if __name__ == '__main__':
    vistog = False

    ARGBASE = ['-growthRate']

    ARGS = ARGBASE + ["%1.2f" % 0.01]
    ran1_MM_gL = Simulation([1.], np.arange(0, 102.5, 2.5), 'evolran1_gL', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.14]
    ran1_MM_gM = Simulation([1.], np.arange(0, 24, 3), 'evolran1_gM', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.30]
    ran1_MM_gH = Simulation([1.], np.arange(0, 24, 3), 'evolran1_gH', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.01]
    ran2_MM_gL = Simulation([2.], np.arange(0, 102.5, 2.5), 'evolran2_gL', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.14]
    ran2_MM_gM = Simulation([2.], np.arange(0, 105, 5), 'evolran2_gM', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.30]
    ran2_MM_gH = Simulation([2.], np.arange(0, 105, 5), 'evolran2_gH', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.01]
    ran0_MM_gL = Simulation([0.25], np.arange(0, 102.5, 2.5), 'evolran0_gL', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.14]
    ran0_MM_gM = Simulation([0.25], np.arange(0, 105, 5), 'evolran0_gM', 5, ARGS)

    ARGS = ARGBASE + ["%1.2f" % 0.30]
    ran0_MM_gH = Simulation([0.25], np.arange(0, 105, 5), 'evolran0_gH', 5, ARGS)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    ran2_MM_gL.sep_visualize(ax1)
    ran2_MM_gM.sep_visualize(ax2)
    ran2_MM_gH.sep_visualize(ax3)
    ax1.set_title('Competition verification, growth rate = 0.01')
    ax2.set_title('Competition verification, growth rate = 0.14')
    ax3.set_title('Competition verification, growth rate = 0.30')
    plt.show()

    if vistog:
        #ran1_MM_gL.visualize()
        ran1_MM_gM.visualize()
        #ran1_MM_gH.visualize()