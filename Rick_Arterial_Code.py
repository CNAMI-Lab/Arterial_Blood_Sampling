#Import Modules#
import os
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from tkinter import Tk   
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

class scan:
        def __init__(self):
            messagebox.showinfo("Information","Select Counter Sheet")
            self.df_curve = pd.read_excel(askopenfilename())
            messagebox.showinfo("Information","Select Timing Sheet")
            self.df_timing = pd.read_excel(askopenfilename())
            self.date = datetime.fromisoformat(input("Enter Date of Scan (YYYY-MM-DD):"))
            self.tracer = str(input("Enter Radiotracer (F-18):"))
            if "F-18" in self.tracer:
               self.tracer_hl = 109.771
            elif "C-11" in self.tracer:
               self.tracer_hl = 20.38


def dataClean(scan):
    #Remove NaN

    df_curve = scan.df_curve
    df_timing = scan.df_timing

    df_timing = df_timing.dropna(axis=0, how ='all')
    df_nan_trial  = df_timing[df_timing.isnull().sum(axis=1) <7]

    #Create Arrays
    array_trial = np.array(df_nan_trial)
    array_curve = np.array(df_curve)

    #Remove rows with unnecessary info
    dataframe_timepoint = pd.DataFrame(array_trial[min(np.where(array_trial=='Sample')[0]):])
    dataframe_timepoint = dataframe_timepoint.rename(columns= dataframe_timepoint.iloc[0]).drop(dataframe_timepoint.index[0])
    dataframe_timepoint = dataframe_timepoint[pd.to_numeric(dataframe_timepoint['Sample'], errors='coerce').notnull()]  

    dataframe_curve = pd.DataFrame(array_curve[min(np.where(array_curve=='POS')[0]):]) 
    dataframe_curve = dataframe_curve.rename(columns=dataframe_curve.iloc[0]).drop(dataframe_curve.index[0])
    dataframe_curve = dataframe_curve.iloc[:,:5]

    #Add BARCODE
    blood_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and 'lood' in col]
    plasma_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and 'lasma' in col]
    if(len(blood_col)>1):
        blood_col = [match for match in blood_col if "GC" in match]
    if(len(plasma_col)>1):
        plasma_col = [match for match in plasma_col if "GC" in match]
    start_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and (r'Draw Time start') in col]
    finish_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and (r'Draw Time finish') in col]
    dataframe_curve['BARCODE  '] = ""
    dataframe_curve['Draw Start'] = ""
    dataframe_curve['Draw Finish'] = ""
    for i in range(dataframe_timepoint.shape[0]):
        dataframe_curve['BARCODE  '][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] = "WB " + f"{i+1:02}" 
        dataframe_curve['BARCODE  '][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[plasma_col].iloc[i]))[0])+1] = "PL " + f"{i+1:02}" 
        
        dataframe_curve['Draw Start'][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[plasma_col].iloc[i]))[0])+1] =  dataframe_timepoint[start_col].iloc[i]
        dataframe_curve['Draw Finish'][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[plasma_col].iloc[i]))[0])+1] = dataframe_timepoint[finish_col].iloc[i]
        dataframe_curve['Draw Start'][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] =  dataframe_timepoint[start_col].iloc[i]
        dataframe_curve['Draw Finish'][int(np.where(dataframe_curve["POS"]==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] = dataframe_timepoint[finish_col].iloc[i]
        #not sure if +1 works all the time

    #Remove Rct's if necessary
    recounts = dataframe_curve[dataframe_curve['BARCODE  '].str.contains('rct')==True]
    if len(recounts>0): 
        recounts = pd.Series(recounts['BARCODE  '].str.slice(0,5)).drop_duplicates(keep="first",inplace=False)
        for i in recounts:
            all_indices = dataframe_curve[dataframe_curve['BARCODE  '].str.contains(i)==True]['ELTIME']
            max_index = pd.Series.idxmax(pd.to_numeric(all_indices))
            all_indices = all_indices.drop(max_index)
            dataframe_curve = dataframe_curve.drop(all_indices.index)

    scan.dataframe_curve = dataframe_curve
    scan.dataframe_timepoint = dataframe_timepoint
    scan.array_trial = array_trial
    scan.df_timing = df_timing
    scan.df_nan_trial = df_nan_trial

    return scan

def collectTimes(scan):
    #finish_index = np.where(array_trial=='Finish inj (clock):')
    #finish_time = array_trial[finish_index[0],finish_index[1]+1][0]

    start_index = np.where(scan.array_trial=='Start Inj (clock):')     #is capital "I" going to be consistent
    if np.asarray(start_index).size == 0:
        start_index = scan.df_timing.columns.get_loc([col for col in scan.df_timing.columns if isinstance(col, str) and 'Start Inj (clock):' in col][0])
        start_time = scan.df_timing.columns[start_index+1]
    else:
        start_time = scan.array_trial[start_index[0],start_index[1]+1][0]
    scan.start_time = start_time
    return scan
    #add check that length of trial is plausible?

def determineDuration(scan):
    gc_start_idx = list(np.where(scan.df_timing=='GC Start time:'))
    gc_start_idx[1] = gc_start_idx[1]+1
    gc_start = scan.df_nan_trial.iloc[gc_start_idx[0],gc_start_idx[1]]
    gc_start = datetime.time(datetime.strptime(pd.DataFrame.to_string(gc_start, header=None, index=None)," %H:%M:%S")) # this time needs to be consistent
    duration = datetime.combine(scan.date.min, gc_start) - datetime.combine(scan.date.min, scan.start_time)
    minute = np.timedelta64(duration, 'm')
    durminutes = minute.astype('timedelta64[m]')
    scan.duration_in_minutes = durminutes/np.timedelta64(1, 'm')
    return scan

def calculateBackground(scan):
    CHsheet = []
    CPMsheet = []
    BarcodeSheet = []
    PositionSheet = []
    #loop through CPM and find Background Noise
    Gamma_Counter_Sheet = np.array(scan.dataframe_curve)
    for i in range(2, scan.dataframe_curve.shape[0]-2):
        if Gamma_Counter_Sheet[i,1] == 1 and (int(Gamma_Counter_Sheet[i-2,2] ) + int(Gamma_Counter_Sheet[i-1,2] )) < 100 and (int(Gamma_Counter_Sheet[i+2,2] )+int(Gamma_Counter_Sheet[i+3,2] ))  < 100 and (int(Gamma_Counter_Sheet[i,2] )+int(Gamma_Counter_Sheet[i+1,2] ))  < 100 :
                    CPMsheet = np.concatenate((CPMsheet,Gamma_Counter_Sheet[i:i+2,2]))
                    CHsheet = np.concatenate((CHsheet,Gamma_Counter_Sheet[i:i+2,1]))
                    BarcodeSheet = np.concatenate((BarcodeSheet,Gamma_Counter_Sheet[i:i+2,5]))
                    PositionSheet = np.concatenate((PositionSheet,Gamma_Counter_Sheet[i:i+2,0]))
    len(PositionSheet)
    d = { 'POS' : PositionSheet,'CH' : CHsheet, 'CPM' : CPMsheet, 'Barcode' : BarcodeSheet}
    scan.Background = pd.DataFrame(d)
    scan.Gamma_Counter_Sheet = Gamma_Counter_Sheet

    #calculate background value
    scan.bkg_value = 2*(sum(pd.to_numeric(scan.Background['CPM']))/len(scan.Background))
    return scan

def calculateAvgActLeft(scan):
    sc = 0
    al = 0
    if len([i for i, x in enumerate(scan.df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)NIST').any(), axis=1)) if x]) > 0:
        std_idx_n = [i for i, x in enumerate(scan.df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)NIST').any(), axis=1)) if x][0]
        sc += 1
        n_standard = int(scan.df_curve.iloc[std_idx_n,2] ) + int(scan.df_curve.iloc[std_idx_n+1,2] )-scan.bkg_value
        n_std_date = datetime(2021,11,10 ) #replace hard code with lookup/input
        n_std_activity = 0.111  #replace hard code with lookup/input
        n_std_time_elapsed = scan.date - n_std_date
        n_std_time_elapsed = n_std_time_elapsed.days
        n_activity_left = (n_std_activity)*(2**(-(n_std_time_elapsed)/270.8))
        act_per_act_left_n = n_standard/n_activity_left
        al += act_per_act_left_n

    if len([i for i, x in enumerate(scan.df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)CCIR').any(), axis=1)) if x]) > 0:
        std_idx_c = [i for i, x in enumerate(scan.df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)CCIR').any(), axis=1)) if x][0]
        sc += 1
        c_standard = int(scan.df_curve.iloc[std_idx_c,2] ) + int(scan.df_curve.iloc[std_idx_c+1,2] )-scan.bkg_value
        c_std_date = datetime(2022,2,1) #replace hard code with lookup/input
        c_std_activity = 0.1052  #replace hard code with lookup/input
        c_std_time_elapsed = scan.date - c_std_date
        c_std_time_elapsed = c_std_time_elapsed.days
        c_activity_left = (c_std_activity)*(2**(-(c_std_time_elapsed)/270.8))
        act_per_act_left_c = c_standard/c_activity_left
        al += act_per_act_left_c

    #if len([i for i, x in enumerate(df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)old').any(), axis=1)) if x]) > 0:
    #  std_idx_o = [i for i, x in enumerate(df_curve.apply(lambda row: row.astype(str).str.contains(r'(?i)old').any(), axis=1)) if x][0]
    #  sc += 1
    #  old_standard = int(df_curve.iloc[std_idx_o,2] ) + int(df_curve.iloc[std_idx_o+1,2] )-bkg_value
    #  old_std_date = datetime(2020,3,13) #replace hard code with lookup/input
    #  old_std_activity = 0.141 #replace hard code with lookup/input
    #  old_std_time_elapsed = date - old_std_date 
    #  old_std_time_elapsed = old_std_time_elapsed.days
    #  old_activity_left = (old_std_activity)*(2**(-(old_std_time_elapsed)/270.8))
    #  act_per_act_left_old = old_standard/old_activity_left
    #  al += act_per_act_left_old

    scan.avg_act_per_act_left = al/sc #add check that the date isn't more than 271 days old and that value isn't far from 720,000
    return(scan)

def calculateMuCi(scan):
    muCi_calculator_FINAL_sheet = []
    muCi_FINAL_sheet = []
    BarcodeSheet = []
    PositionSheet = []
    CH_Sheet = []
    CPM_Sheet = []
    kBq_Sheet = []
    start_time_Sheet = []
    stop_time_Sheet = []

    for i in range(0, scan.Gamma_Counter_Sheet.shape[0]-1):
        if scan.Gamma_Counter_Sheet[i,1] == 1:
            muci_calculator = (int(scan.Gamma_Counter_Sheet[i,2] )+ int(scan.Gamma_Counter_Sheet[i+1,2] ) - scan.bkg_value)*(2**((scan.duration_in_minutes+float(scan.Gamma_Counter_Sheet[i,4]))/scan.tracer_hl))
            muci = muci_calculator/scan.avg_act_per_act_left
            if 'PL' in scan.Gamma_Counter_Sheet[i,5]:
                muci = muci/0.4
            kBq = muci * 37
            kBq_Sheet.append(kBq)
            if(isinstance(scan.Gamma_Counter_Sheet[i,6], pd.Series)):
                start_time_Sheet.append(scan.Gamma_Counter_Sheet[i,6][0])
                stop_time_Sheet.append(scan.Gamma_Counter_Sheet[i,7][0])
            else:
                start_time_Sheet.append(scan.Gamma_Counter_Sheet[i,6])
                stop_time_Sheet.append(scan.Gamma_Counter_Sheet[i,7])
            muCi_calculator_FINAL_sheet.append(muci_calculator)
            muCi_FINAL_sheet.append(muci)
            CH_Sheet.append(scan.Gamma_Counter_Sheet[i:i+1,1])
            BarcodeSheet.append(scan.Gamma_Counter_Sheet[i,5])
            PositionSheet.append(scan.Gamma_Counter_Sheet[i,0])
            CPM_Sheet.append(scan.Gamma_Counter_Sheet[i:i+1,2])

    d_final = { 'POS' : PositionSheet, 'Barcode' : BarcodeSheet, 'draw start': start_time_Sheet, 'draw finish': stop_time_Sheet, 'muci-calculator' : muCi_calculator_FINAL_sheet, 'muci ': muCi_FINAL_sheet, 'kBq': kBq_Sheet}
    final_calculation = pd.DataFrame(d_final)
    final_calculation = final_calculation.iloc[[i for i, x in enumerate(final_calculation.apply(lambda row: row.astype(str).str.contains(r'[a-zA-Z]{2} \d{2}').any(), axis=1)) if x],]
    scan.final_calculation = final_calculation.sort_values(by='Barcode')
    return scan

def outputData(scan):
    pd.DataFrame(scan.final_calculation).to_csv(str(str(scan.date)[:10]+'_Final Calculation.csv'), index = False)

scan1 = scan() 
scan1 = dataClean(scan1)
scan1 = collectTimes(scan1)
scan1 = determineDuration(scan1)
scan1 = calculateBackground(scan1)
scan1 = calculateAvgActLeft(scan1)
scan1 = calculateMuCi(scan1)
outputData(scan1)


