import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def movmean_data(A, k):
    """
    Args:
        A: DataFrame or Series 
        k: window size (k)
    """
    x = A.rolling(k, min_periods=1, center=True).mean().to_numpy()
    return x

def movmedian_data(A, k):
    """
    Args:
        A: DataFrame or Series 
        k: window size (k)
    """
    x = A.rolling(k, min_periods=1, center=True).median().to_numpy()
    return x

def calculate_heart_rate(data, fs=250):
    """
    Calculate average heart rate from signal data based on peaks.
    """
    ampl_, __ = find_peaks(data, distance=int(0.5 * fs))
    if len(ampl_) < 2:
        return np.nan
    RR = ampl_[1:] - ampl_[:-1]
    FHR = 60 * fs / RR
    FHR_average = np.mean(FHR)
    return FHR_average

def read_ppg_file(file_name):
    """
    Returns:
        ppg_red_data: List of PPG RED data
        ppg_ir_data: List of PPG IR data
    """
    ppg_red_data = []
    ppg_ir_data = []
    
    df_data = pd.read_csv(file_name)
    ppg_red_data = list(df_data['RED'].values)
    ppg_ir_data = list(df_data['IR'].values)

    return ppg_red_data, ppg_ir_data

def visualize_and_process(file_name, visualize = True):
    # ========== CONFIGURATION ==============
    fs = 250
    windowsize = int(0.1 * fs)

    # 1. Read data
    ppg_red_data, ppg_ir_data = read_ppg_file(file_name)
    indices = np.arange(len(ppg_red_data))

    # 2. Plot raw PPG IR và RED
    if visualize:
        fig, axs = plt.subplots(2, 1, sharex=True, num="PPG Raw Signals")
        axs[0].plot(indices, ppg_red_data)
        axs[0].set_title("PPG RED Data")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("ADC value")
        axs[1].plot(indices, ppg_ir_data)
        axs[1].set_title("PPG IR Data")
        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("ADC value")
        plt.tight_layout()
        plt.show()

    # 3. Filtering (Median and Moving Average)
    red_median = movmedian_data(pd.DataFrame(ppg_red_data), windowsize).flatten()
    ir_median = movmedian_data(pd.DataFrame(ppg_ir_data), windowsize).flatten()
    red_movmean = movmean_data(pd.DataFrame(red_median), fs).flatten()
    ir_movmean = movmean_data(pd.DataFrame(ir_median), fs).flatten()

    # 4. Signal after filtering
    if visualize: 
        fig, axs = plt.subplots(2, 1, sharex=True, num="PPG Filtered")
        axs[0].plot(indices, red_median, label="Median")
        axs[0].plot(indices, red_movmean, label="MovMean")
        axs[0].set_title("PPG RED Filtered")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("ADC value")
        axs[0].legend()
        axs[1].plot(indices, ir_median, label="Median")
        axs[1].plot(indices, ir_movmean, label="MovMean")
        axs[1].set_title("PPG IR Filtered")
        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("ADC value")
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    # 5. Find peaks of filtered signals
    ampl_red, _ = find_peaks(red_median, distance=int(0.5 * fs))
    ampl_ir, _ = find_peaks(ir_median, distance=int(0.5 * fs))

    # 6. Heart rate rolling
    heart_rate_red = pd.DataFrame(red_median).rolling(window=400, min_periods=1, center=True).apply(lambda x: calculate_heart_rate(x), raw=True)
    heart_rate_ir = pd.DataFrame(ir_median).rolling(window=400, min_periods=1, center=True).apply(lambda x: calculate_heart_rate(x), raw=True)
    indices_heart_rate = np.arange(len(heart_rate_red))
    if visualize:
        fig, axs = plt.subplots(2, 1, sharex=True, num="Heart Rate Rolling")
        axs[0].plot(indices_heart_rate, heart_rate_red)
        axs[0].set_title("Rolling Heart Rate (RED)")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("Heart Rate")
        axs[1].plot(indices_heart_rate, heart_rate_ir)
        axs[1].set_title("Rolling Heart Rate (IR)")
        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("Heart Rate")
        plt.tight_layout()
        plt.show()

    # 7. AC extraction
    ac_red = np.array(ppg_red_data) - red_movmean
    ac_red_invert = -ac_red
    ac_ir = np.array(ppg_ir_data) - ir_movmean
    ac_ir_invert = -ac_ir

    ac_red_median = movmedian_data(pd.DataFrame(ac_red), windowsize).flatten()
    ac_red_invert_median = movmedian_data(pd.DataFrame(ac_red_invert), windowsize).flatten()
    ac_ir_median = movmedian_data(pd.DataFrame(ac_ir), windowsize).flatten()
    ac_ir_invert_median = movmedian_data(pd.DataFrame(ac_ir_invert), windowsize).flatten()

    # 8. Find peaks in AC signals, return indices of peaks 
    ampl_ac_red, _ = find_peaks(ac_red_median, distance=int(0.3 * fs), width=0.2 * fs)
    ampl_ac_red_invert, _ = find_peaks(ac_red_invert_median, distance=int(0.15 * fs), width=0.15 * fs)
    ampl_ac_ir, _ = find_peaks(ac_ir_median, distance=int(0.3 * fs), width=0.19 * fs)
    ampl_ac_ir_invert, _ = find_peaks(ac_ir_invert_median, distance=int(0.15 * fs), width=0.11 * fs)

    # 9. Plot AC signals with peaks
    if visualize:
        fig, axs = plt.subplots(2, 1, sharex=True, num="AC Signals with Peaks")
        axs[0].plot(indices, ac_red_median, label="AC RED")
        axs[0].plot(ampl_ac_red, ac_red_median[ampl_ac_red], "r*", label="Peaks")
        axs[0].set_title("AC RED Median + Peaks")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("ADC value")
        axs[0].legend()
        axs[1].plot(indices, ac_ir_median, label="AC IR")
        axs[1].plot(ampl_ac_ir, ac_ir_median[ampl_ac_ir], "r*", label="Peaks")
        axs[1].set_title("AC IR Median + Peaks")
        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("ADC value")
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    # 10. Ghép strip AC
    ac_red_copy = ampl_ac_red.copy()
    ac_red_invert_copy = ampl_ac_red_invert.copy()
    ac_ir_copy = ampl_ac_ir.copy()
    ac_ir_invert_copy = ampl_ac_ir_invert.copy()
    if ac_red_copy[0] > ac_red_invert_copy[0]:
        ac_red_invert_copy = ac_red_invert_copy[1:]
    if ac_red_copy[-1] > ac_red_invert_copy[-1]:
        ac_red_copy = ac_red_copy[:len(ac_red_copy) - 1]
    if ac_ir_copy[0] > ac_ir_invert_copy[0]:
        ac_ir_invert_copy = ac_ir_invert_copy[1:]
    if ac_ir_copy[-1] > ac_ir_invert_copy[-1]:
        ac_ir_copy = ac_ir_copy[:len(ac_ir_copy) - 1]

    ac_strip_red = ac_red_median[ac_red_copy] + ac_red_invert_median[ac_red_invert_copy]
    ac_strip_ir = ac_ir_median[ac_ir_copy] + ac_ir_invert_median[ac_ir_invert_copy]

    # 11. Min DC
    min_dc_red = [np.min(red_movmean[ac_red_copy[i]:ac_red_invert_copy[i]+1]) for i in range(len(ac_red_copy))]
    min_dc_ir = [np.min(ir_movmean[ac_ir_copy[i]:ac_ir_invert_copy[i]+1]) for i in range(len(ac_ir_copy))]

    # 12. SpO2 và ROR
    ac_div_dc_red = ac_strip_red / min_dc_red
    ac_div_dc_ir = ac_strip_ir / min_dc_ir
    ror = ac_div_dc_ir / ac_div_dc_red
    spo2 = 110 - 25 * ror
    indices_spo2 = np.arange(len(spo2))

    # 13. Plot results
    if visualize:
        fig, axs = plt.subplots(2, 2, sharex=True, num="AC/DC and SpO2")
        axs[0, 0].plot(indices_spo2, ac_strip_ir)
        axs[0, 0].set_title("ac ir")
        axs[0, 1].plot(indices_spo2, ac_strip_red)
        axs[0, 1].set_title("ac red")
        axs[1, 0].plot(indices_spo2, min_dc_ir)
        axs[1, 0].set_title("dc ir")
        axs[1, 1].plot(indices_spo2, min_dc_red)
        axs[1, 1].set_title("dc red")
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 1, sharex=True, num="AC/DC Ratio")
        axs[0].plot(indices_spo2, ac_div_dc_ir)
        axs[0].set_title("ac ir / dc ir")
        axs[1].plot(indices_spo2, ac_div_dc_red)
        axs[1].set_title("ac red / dc red")
        plt.tight_layout()
        plt.show()

    plt.figure("ROR")
    plt.plot(indices_spo2, ror)
    plt.title(f'ROR plot with average value {np.mean(ror)}')
    plt.tight_layout()
    plt.show()

    plt.figure("SpO2")
    plt.plot(indices_spo2, spo2)
    plt.xlabel("So mau")
    plt.ylabel("% SpO2")
    plt.title(f'SpO2, Average SpO2 = {np.mean(spo2):.2f}, Heart rate = {calculate_heart_rate(red_median):.2f}')
    plt.grid()
    plt.show()