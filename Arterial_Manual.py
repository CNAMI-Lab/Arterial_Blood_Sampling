import importlib
import pandas as pd
import sys
#sys.path.insert(0, '/Users/r.reneau/Downloads/')
sys.path.insert(0, 'C:/Users/rickr/Downloads/')
import os
from datetime import datetime
import Rick_Arterial_Code as acode
importlib.reload(acode)
scan6 = acode.scan()
scan6.df_timing = pd.read_excel('Timings_GC_pos_RO948_2023_03_30.xlsx')
scan6.df_curve = pd.read_excel('2023.03.30.RO948.xlsx')
scan6.date = datetime.fromisoformat("2023-03-30")
scan6.tracer = "F-18"
scan6.tracer_hl = 109.771
scan6.tracer_name = "RO948"
scan6 = acode.dataClean(scan6)
scan6 = acode.collectTimes(scan6)
scan6 = acode.determineDuration(scan6)
scan6 = acode.calculateBackground(scan6)
scan6.avg_act_per_act_left = 717503.9324606133
scan6 = acode.calculateMuCi(scan6)
acode.outputData(scan6)
hplc_dir = "C:/Users/rickr/Downloads/2023_03_30_18F_RO948"
hplc_dataframes = []
hplc_dataframes = acode.read_metabolites(hplc_dir,hplc_dataframes)
hplc_data = acode.extractMetaboliteData(hplc_dataframes)
tracer_hl = 109.771
tracer_name = "RO948"
hplc_corrections = acode.correctMetabolites(hplc_data,tracer_hl,tracer_name)
pc_corrected = acode.correctPlasma(hplc_corrections,scan6.final_calculation,scan6.start_time)
pd.DataFrame(pc_corrected).to_csv(str(str(scan6.date)[:10]+'_'+tracer_name+'_Metabolite_Corrected_Plasma_Curve.csv'), index = False)
