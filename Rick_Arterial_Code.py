#Import Modules#
import os
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import time
from tkinter import Tk   
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

class scan:
    pass

def collectData():
    scan1 = scan()
    messagebox.showinfo("Information","Select Counter Sheet")
    scan1.df_curve = pd.read_excel(askopenfilename())
    messagebox.showinfo("Information","Select Timing Sheet")
    scan1.df_timing = pd.read_excel(askopenfilename())
    scan1.date = datetime.fromisoformat(input("Enter Date of Scan (YYYY-MM-DD):"))
    scan1.tracer = str(input("Enter Radiotracer (F-18):"))
    if "F-18" in scan1.tracer:
        scan1.tracer_hl = 109.771
    elif "C-11" in scan1.tracer:
        scan1.tracer_hl = 20.38
    return scan


def dataClean(scan):
    #Remove NaN

    df_curve = scan.df_curve
    df_timing = scan.df_timing

    df_timing = df_timing.dropna(axis=0, how ='all') #not sure if necessary
    df_nan_trial  = df_timing[df_timing.isnull().sum(axis=1) <7] #not sure if necessary

    #Create Arrays
    array_trial = np.array(df_nan_trial)
    array_curve = np.array(df_curve)

    #Remove rows with unnecessary info
    dataframe_timepoint = pd.DataFrame(array_trial[min(np.where(array_trial=='Sample')[0]):]) #removes all rows before "Sample"
    dataframe_timepoint = dataframe_timepoint.rename(columns= dataframe_timepoint.iloc[0]).drop(dataframe_timepoint.index[0]) #sets column names to first row and drops first row
    dataframe_timepoint = dataframe_timepoint[pd.to_numeric(dataframe_timepoint['Sample'], errors='coerce').notnull()]   #removes all rows that don't have a number in the "Sample" column

    dataframe_curve = pd.DataFrame(array_curve[min(np.where(array_curve=='POS')[0]):]) #removes all rows before "POS"
    dataframe_curve = dataframe_curve.rename(columns=dataframe_curve.iloc[0]).drop(dataframe_curve.index[0]) #sets column names to first row and drops first row
    #dataframe_curve = dataframe_curve.iloc[:,:5] #removes all columns after "CPM"

    #Add BARCODE
    blood_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and 'lood' in col]
    plasma_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and 'lasma' in col]
    if(len(blood_col)>1):
        blood_col = [match for match in blood_col if "GC" in match]
    if(len(plasma_col)>1):
        plasma_col = [match for match in plasma_col if "GC" in match]
    start_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and (r'start') in col]
    finish_col = [col for col in dataframe_timepoint.columns if isinstance(col, str) and (r'finish') in col]
    #if dataframe_timepoint's blood_col contains a number then run this code
    #if any(char.isdigit() for value in dataframe_timepoint[blood_col] for char in str(value)):
    dataframe_curve['BARCODE  '] = ""
    dataframe_curve['Draw Start'] = ""
    dataframe_curve['Draw Finish'] = ""
    barcode_col_idx = dataframe_curve.columns.get_loc([col for col in dataframe_curve.columns if isinstance(col, str) and 'BARCODE' in col][0])
    for i in range(dataframe_timepoint.shape[0]):
            # dataframe_curve['BARCODE  '][int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] = "WB " + f"{i+1:02}" 
            # dataframe_curve['BARCODE  '][int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][-1])))[0])+1] = "PL " + f"{i+1:02}" 
            
            # dataframe_curve['Draw Start'][int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][-1])))[0])+1] =  dataframe_timepoint[start_col].iloc[i]
            # dataframe_curve['Draw Finish'][int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][-1])))[0])+1] = dataframe_timepoint[finish_col].iloc[i]
            # dataframe_curve['Draw Start'][int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] =  dataframe_timepoint[start_col].iloc[i]
            # dataframe_curve['Draw Finish'][int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0])+1] = dataframe_timepoint[finish_col].iloc[i]
            #not sure if +1 works all the time
            dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0]),barcode_col_idx] = "WB " + f"{i+1:02}" 
            dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][0])))[0]),barcode_col_idx] = "PL " + f"{i+1:02}" 
            if isinstance(dataframe_timepoint[start_col].iloc[i][0], time):
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0]),barcode_col_idx+1] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0]),barcode_col_idx+2] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][0])))[0]),barcode_col_idx+1] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(last_numbers(dataframe_timepoint[plasma_col].iloc[i][0])))[0]),barcode_col_idx+2] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%H:%M:%S"))
            else:
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0]),barcode_col_idx+1] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[blood_col].iloc[i]))[0]),barcode_col_idx+2] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[plasma_col].iloc[i]))[0]),barcode_col_idx+1] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
                dataframe_curve.iloc[int(np.where(dataframe_curve["POS"].astype('float32')==int(dataframe_timepoint[plasma_col].iloc[i]))[0]),barcode_col_idx+2] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))

    # else:
    #     dataframe_curve.columns.values[6] = 'BARCODE  '
    #     dataframe_curve['Draw Start'] = ""
    #     dataframe_curve['Draw Finish'] = ""
    #     for i in range(dataframe_timepoint.shape[0]):
    #         dataframe_curve.iloc[np.where(dataframe_curve['BARCODE  '].str.contains(f"WB {i+1:02}") & dataframe_curve['BARCODE  '].notna())[0][0],8] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
    #         dataframe_curve.iloc[np.where(dataframe_curve['BARCODE  '].str.contains(f"WB {i+1:02}") & dataframe_curve['BARCODE  '].notna())[0][0],9] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
    #         dataframe_curve.iloc[np.where(dataframe_curve['BARCODE  '].str.contains(f"PL {i+1:02}") & dataframe_curve['BARCODE  '].notna())[0][0],8] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[start_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
    #         dataframe_curve.iloc[np.where(dataframe_curve['BARCODE  '].str.contains(f"PL {i+1:02}") & dataframe_curve['BARCODE  '].notna())[0][0],9] =  datetime.time(datetime.strptime(''.join(dataframe_timepoint[finish_col].iloc[i].astype(str)), "%Y-%m-%d %H:%M:%S"))
        

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
    scan.barcode_col_idx = barcode_col_idx

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
    if isinstance(start_time, datetime):
        start_time = start_time.time()
    scan.start_time = start_time
    return scan
    #add check that length of trial is plausible?

def determineDuration(scan):
    gc_start_col = [col for col in scan.df_curve.columns if isinstance(col, str) and 'CCIR' in col]
    gc_start = 0
    for index, value in scan.df_curve[gc_start_col].items():
        for element in value:
            if isinstance(element, time):
                gc_start = element
                break
    #if gc_start is a pd.Series then convert to datetime.time
    if isinstance(gc_start, pd.Series):
        gc_start = datetime.time(datetime.strptime(''.join(gc_start.astype(str)),"%H:%M:%S")) # this time needs to be consistent
    if gc_start == 0:
        gc_start_idx = list(np.where(scan.df_timing=='GC Start time:'))
        gc_start_idx[1] = gc_start_idx[1]+1
        gc_start = scan.df_nan_trial.iloc[gc_start_idx[0],gc_start_idx[1]].iloc[0,0]
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
    if sc == 0:
        scan.avg_act_per_act_left = 720000
    else:
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
            if 'PL' in scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx ]:
                muci = muci/0.4   #add check to make sure 1ml and .4ml are standard in timing sheet columns E and G
            kBq = muci * 37
            kBq_Sheet.append(kBq)
            if(isinstance(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx + 1], pd.Series)):
                start_time_Sheet.append(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx +1][0])
                stop_time_Sheet.append(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx +2][0])
            else:
                start_time_Sheet.append(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx +1])
                stop_time_Sheet.append(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx +2])
            muCi_calculator_FINAL_sheet.append(muci_calculator)
            muCi_FINAL_sheet.append(muci)
            CH_Sheet.append(scan.Gamma_Counter_Sheet[i:i+1,1])
            BarcodeSheet.append(scan.Gamma_Counter_Sheet[i,scan.barcode_col_idx])
            PositionSheet.append(scan.Gamma_Counter_Sheet[i,0])
            CPM_Sheet.append(scan.Gamma_Counter_Sheet[i:i+1,2])

    d_final = { 'POS' : PositionSheet, 'Barcode' : BarcodeSheet, 'draw start': start_time_Sheet, 'draw finish': stop_time_Sheet, 'muci-calculator' : muCi_calculator_FINAL_sheet, 'muci ': muCi_FINAL_sheet, 'kBq': kBq_Sheet}
    final_calculation = pd.DataFrame(d_final)
    final_calculation = final_calculation.iloc[[i for i, x in enumerate(final_calculation.apply(lambda row: row.astype(str).str.contains(r'[a-zA-Z]{2} \d{2}').any(), axis=1)) if x],]
    if(isinstance(final_calculation.iloc[1,2],time)):
        for i in range(0, final_calculation.shape[0]):
            final_calculation.iloc[i,2] = datetime.combine(scan.date.date(),final_calculation.iloc[i,2])
            final_calculation.iloc[i,3] = datetime.combine(scan.date.date(),final_calculation.iloc[i,3])
    scan.final_calculation = final_calculation.sort_values(by='Barcode')
    return scan

def outputData(scan):
    pd.DataFrame(scan.final_calculation).to_csv('C:/Users/rickr/Downloads/'+str(str(scan.date)[:10]+'_Final Calculation.csv'), index = False)

def last_numbers(s):
    return int(s.split(',')[-1].strip()) if isinstance(s, str) else s

def read_metabolites(hplc_dir,dataframes):
    for filename in os.listdir(hplc_dir):
        if filename.endswith(".xls"):
            file_path = os.path.join(hplc_dir, filename)
            df = pd.read_excel(file_path)
            dataframes.append(df)
    return dataframes

def extractMetaboliteData(dataframes):
    # create an empty dataframe to store the extracted data
    final_df = pd.DataFrame(columns=["Name", "RT 1", "Area 1", "RT 2", "Area 2", "RT 3", "Area 3"])

    # loop through the dataframes and extract the required data
    for df in dataframes:
        # extract name, area, and RT data
        name = df.iloc[1, 2].split()[-2:]
        name = ' '.join(name)
        RT_1 = df.iloc[9, 1]
        area_1 = df.iloc[9, 4]
        RT_2 = df.iloc[10, 1]
        area_2 = df.iloc[10, 4]
        RT_3 = df.iloc[11, 1]
        area_3 = df.iloc[11, 4]
        
        # create a new dataframe with the extracted data
        new_df = pd.DataFrame({"Name": [name], "RT 1": [RT_1], "Area 1": [area_1],
                            "RT 2": [RT_2], "Area 2": [area_2], "RT 3": [RT_3], "Area 3": [area_3]})


        # check if name ends with "re"
        if name.endswith("re"):
            # update the rows with the matching name
            name = df.iloc[1, 2].split()[-3:-1]
            name = ' '.join(name)
            final_df.loc[final_df['Name'] == name, ['RT 1', 'Area 1', 'RT 2', 'Area 2', 'RT 3', 'Area 3']] = [RT_1, area_1, RT_2, area_2, RT_3, area_3]
        elif name.endswith("Blank"):
            pass
        else:
            # append the new dataframe to the final dataframe
            final_df = pd.concat([final_df, new_df], ignore_index=True)

    return final_df

def correctMetabolites(hdf,tracer_hl,tracer_name,date):
    Flow1 = 2
    Flow2 = 2
    if tracer_name == "ASEM":
        Flow2 = 1.5
    elif tracer_name == "MK6240":
        Flow2 = 1.7
    elif tracer_name == "RO948":
        Flow2 = 1.3
    if date > datetime.fromisoformat("2023-07-19"):
        Flow2 = 1.0
    in_vitro_column = hdf["Name"].str.contains(r'\b0 min\b|in vitro|In vitro')
    dose_column = hdf["Name"].str.contains("dose")
    hdf.replace('n.a.', np.nan, inplace=True)
    hdf["Flow 1"] = hdf["Area 1"]*Flow1*2**(hdf["RT 1"]/tracer_hl)
    hdf["Flow 2"] = hdf["Area 2"]*Flow2*2**(hdf["RT 2"]/tracer_hl)
    hdf["Flow 3"] = hdf["Area 3"]*Flow2*2**(hdf["RT 3"]/tracer_hl)
    hdf["Flow Sum"] = np.nan
    hdf["Non-Loss Corrected"] = np.nan
    for i in range(hdf.shape[0]):
        if np.isnan(hdf["Flow 3"].iloc[i]):
            hdf["Flow Sum"].iloc[i] = hdf["Flow 1"].iloc[i] + hdf["Flow 2"].iloc[i]
            hdf["Non-Loss Corrected"].iloc[i] = hdf["Flow 2"].iloc[i]/hdf["Flow Sum"].iloc[i]
        else:
            hdf["Flow Sum"].iloc[i] = hdf["Flow 1"].iloc[i] + hdf["Flow 2"].iloc[i] + hdf["Flow 3"].iloc[i]
            hdf["Non-Loss Corrected"].iloc[i] = hdf["Flow 3"].iloc[i]/hdf["Flow Sum"].iloc[i]
    #check if hdf["Non-Loss Corrected"][dose_column] is NaN
    hdf["Non-Loss Corrected"] = hdf["Non-Loss Corrected"].astype(float)
    if np.isnan(hdf["Non-Loss Corrected"][dose_column]).any():
        hdf["Non-Loss Corrected"][dose_column] = 1
    if np.isnan(hdf["Non-Loss Corrected"][in_vitro_column]).any():
        hdf["Non-Loss Corrected"][in_vitro_column] = 1
    #if isinstance(in_vitro_column, pd.Series):
    #    Correction_Factor = 1
    #else:
    Correction_Factor = float(hdf["Non-Loss Corrected"][dose_column])/float(hdf["Non-Loss Corrected"][in_vitro_column])
    hdf["Percent_Intact"] = hdf["Non-Loss Corrected"]*Correction_Factor
    return hdf

def correctMetabolites_Remove_First(hdf,tracer_hl,tracer_name,date):
    Flow1 = 2
    Flow2 = 2
    if tracer_name == "ASEM":
        Flow2 = 1.5
    elif tracer_name == "MK6240":
        Flow2 = 1.7
    elif tracer_name == "RO948":
        Flow2 = 1.3
    if date > datetime.fromisoformat("2023-07-19"):
        Flow2 = 1.0
    in_vitro_column = hdf["Name"].str.contains(r'\b0 min\b|in vitro')
    dose_column = hdf["Name"].str.contains("dose")
    hdf.replace('n.a.', np.nan, inplace=True)
    #hdf["Flow 1"] = hdf["Area 1"]*Flow1*2**(hdf["RT 1"]/tracer_hl)
    hdf["Flow 1"] = 0
    hdf["Flow 2"] = hdf["Area 2"]*Flow2*2**(hdf["RT 2"]/tracer_hl)
    hdf["Flow 3"] = hdf["Area 3"]*Flow2*2**(hdf["RT 3"]/tracer_hl)
    hdf["Flow Sum"] = np.nan
    hdf["Non-Loss Corrected"] = np.nan
    for i in range(hdf.shape[0]):
        if np.isnan(hdf["Flow 3"].iloc[i]):
            hdf["Flow Sum"].iloc[i] = hdf["Flow 1"].iloc[i] + hdf["Flow 2"].iloc[i]
            hdf["Non-Loss Corrected"].iloc[i] = hdf["Flow 2"].iloc[i]/hdf["Flow Sum"].iloc[i]
        else:
            hdf["Flow Sum"].iloc[i] = hdf["Flow 1"].iloc[i] + hdf["Flow 2"].iloc[i] + hdf["Flow 3"].iloc[i]
            hdf["Non-Loss Corrected"].iloc[i] = hdf["Flow 3"].iloc[i]/hdf["Flow Sum"].iloc[i]
    #check if hdf["Non-Loss Corrected"][dose_column] is NaN
    hdf["Non-Loss Corrected"] = hdf["Non-Loss Corrected"].astype(float)
    if np.isnan(hdf["Non-Loss Corrected"][dose_column]).any():
        hdf["Non-Loss Corrected"][dose_column] = 1
    if isinstance(in_vitro_column, pd.Series):
        Correction_Factor = 1
    else:
        Correction_Factor = float(hdf["Non-Loss Corrected"][dose_column])/float(hdf["Non-Loss Corrected"][in_vitro_column])
    hdf["Percent_Intact"] = hdf["Non-Loss Corrected"]*Correction_Factor
    return hdf

def time_difference(start_time, end_time):
    import datetime
    start_datetime = datetime.datetime.combine(datetime.date.today(), start_time)
    end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)
    return (end_datetime - start_datetime).total_seconds()

def average_time(t1_series, t2_series, start_time):
    import datetime

    # Convert t1_series to timestamps if they are datetime objects
    if isinstance(t1_series.iloc[0], datetime.datetime):
        t1_series = t1_series.apply(lambda t: t.time())

    # Convert t2_series to timestamps if they are datetime objects
    if isinstance(t2_series.iloc[0], datetime.datetime):
        t2_series = t2_series.apply(lambda t: t.time())

    # Convert both series to seconds from midnight
    t1_secs = t1_series.apply(lambda t: time_difference(start_time, t))
    t2_secs = t2_series.apply(lambda t: time_difference(start_time, t))

    # Calculate the average time in seconds from midnight
    avg_secs = (t1_secs + t2_secs) / 2

    # Convert the average time back to a datetime.time object
    avg_times = avg_secs.apply(
        lambda s: datetime.time(hour=int(s // 3600), minute=int((s // 60) % 60), second=int(s % 60))
    )

    return avg_times

def correctPlasma(hdf,pc,start_time):
    # calculate the average time and time difference
    pc["time avg"] = average_time(pc["draw start"], pc["draw finish"], start_time)

    # Filter out rows that don't have "min" in their "Name" column
    pc = pc[pc['Barcode'].str.contains('PL')]

    # Extract the time values from the "time avg" column of the pc dataframe
    pc_times = pc['time avg'].apply(lambda x: (x.hour * 3600 + x.minute * 60 + x.second) / 60)

    # Filter out rows that don't have "min" in their "Name" column
    hdf = hdf[hdf['Name'].str.contains('min')]
    # Remove rows with NaN values in the Percent_Intact column
    hdf = hdf.dropna(subset=['Percent_Intact'])
    hdf["times"] = hdf["Name"].apply(lambda x: int(x.split()[0]))
    #sort hdf by times
    hdf = hdf.sort_values(by=['times'])

    # Extract the correction values from the "Percent Intact" column of the hplc_corrections dataframe
    correction_values = hdf['Percent_Intact']

    # Interpolate the correction values for the time points in the pc dataframe
    interpolated_corrections = np.interp(pc_times, hdf["times"], correction_values) #fitting function

    # Apply the correction to the "muci" column of the pc dataframe
    pc['muci_corrected'] = pc['muci '] * interpolated_corrections
    return pc
