import pandas as pd
import sys
#sys.path.insert(0, '/Users/r.reneau/Downloads/')
sys.path.insert(0, 'C:/Users/rickr/Downloads/')
import Rick_Arterial_Code as acode
import os
from datetime import datetime
#os.chdir('/Users/r.reneau/Downloads/')
os.chdir('C:/Users/rickr/Downloads/')

#import importlib
#import Rick_Arterial_Code as acode
#importlib.reload(acode)

#inputs
times = []
activities = []

#outputs
#every second/minute
times2 = []

#Bq/ml
activities2 = [] 

BFGS_times2 = []
BFGS_activities2 = []

def testCurveGeneration(scan1):
    scan1 = acode.dataClean(scan1)
    scan1 = acode.collectTimes(scan1)
    scan1 = acode.determineDuration(scan1)
    scan1 = acode.calculateBackground(scan1)
    scan1 = acode.calculateAvgActLeft(scan1)
    scan1 = acode.calculateMuCi(scan1)
    scan1.final_calculation

def test_metaboliteParentFraction(inputObject):
    #instantiate myobject
    #verifyequals(myobject.times2 == times2)  (matlab) 
    #verifyequals(myobject.activites2 == activities2)
    pass

def test_regression(BFGS_times2,BFGS_activities2):
    #BFGS
    #verifyequals(myobject.BFGS_times2 == BFGS_times2)
    pass


#TEST Scan 1#
#df = pd.read_excel(r"2022 04 01 ASEM Pl and WB and %intact parent.xlsx", sheet_name= 'Sheet1')
df_curve = pd.read_excel('20221130ASEM.xls')
df_timing = pd.read_excel('ASEM_timings_GC_pos_2022_11_30.xlsx')

scan1 = acode.scan()
scan1.df_curve = df_curve
scan1.df_timing = df_timing
scan1.date = datetime.fromisoformat("2022-11-30")
scan1.tracer = "F-18"
scan1.tracer_hl = 109.771

testCurveGeneration(scan1)

#TEST Scan 2#
df_curve = pd.read_excel('20221116 ASEM.xls')
df_timing = pd.read_excel('ASEM_timings_GC_pos_2022_11_16.xlsx')

scan2 = acode.scan()
scan2.df_curve = df_curve
scan2.df_timing = df_timing
scan2.date = datetime.fromisoformat("2022-11-16")
scan2.tracer = "F-18"
scan2.tracer_hl = 109.771

testCurveGeneration(scan2)


#TEST Scan 3#
df_curve = pd.read_excel('18FAZAN_10_25_2022.xls')
df_timing = pd.read_excel('AZAN_timings_GC_positions_2022_10_25.xlsx')

scan3 = acode.scan()
scan3.df_curve = df_curve
scan3.df_timing = df_timing
scan3.date = datetime.fromisoformat("2022-10-25")
scan3.tracer = "F-18"
scan3.tracer_hl = 109.771

testCurveGeneration(scan3)


#TEST Scan 4#
df_curve = pd.read_excel('2023.01.13.ASEM.xls')
df_timing = pd.read_excel('ASEM_timings_GC_pos_2023_01_13.xlsx')

scan4 = acode.scan()
scan4.df_curve = df_curve
scan4.df_timing = df_timing
scan4.date = datetime.fromisoformat("2023-01-13")
scan4.tracer = "F-18"
scan4.tracer_hl = 109.771

testCurveGeneration(scan4)


#2022 09 15 AZAN
#hplc_dir = "C:/Users/rickr/Downloads/20230117_AZAN_met"
hplc_dir = "C:/Users/rickr/Downloads/hplc_20230321"
#hplc_dir = "C:/Users/rickr/Downloads/2023_04_07_18F_ASEM"
#hplc_dir = "C:/Users/rickr/Downloads/2023_03_30_18F_RO948"
hplc_dataframes = []
hplc_dataframes = acode.read_metabolites(hplc_dir,hplc_dataframes)
hplc_data = acode.extractMetaboliteData(hplc_dataframes)
tracer_hl = 109.771
#tracer_name = "RO948"
tracer_name = "ASEM"
hplc_corrections = acode.correctMetabolites(hplc_data,tracer_hl,tracer_name)
