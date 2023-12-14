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
scan6.df_timing = pd.read_excel('C:/Users/rickr/Downloads/20230928 RO-948 GC Timing.xlsx')
scan6.df_curve = pd.read_excel('C:/Users/rickr/Downloads/2023.09.28.RO-948.xls')
scan6.date = datetime.fromisoformat("2023-09-28")
scan6.tracer = "F-18"
scan6.tracer_hl = 109.771
scan6.tracer_name = "RO948"
scan6 = acode.dataClean(scan6)
scan6 = acode.collectTimes(scan6)
scan6 = acode.determineDuration(scan6)
scan6 = acode.calculateBackground(scan6)
scan6 = acode.calculateAvgActLeft(scan6)
scan6 = acode.calculateMuCi(scan6)
acode.outputData(scan6)
hplc_dir = "C:/Users/rickr/Downloads/2023 09 28 18F RO948 - modified data"
hplc_dataframes = []
hplc_dataframes = acode.read_metabolites(hplc_dir,hplc_dataframes)
hplc_data = acode.extractMetaboliteData(hplc_dataframes)
tracer_hl = 109.771
tracer_name = "RO948"
date = datetime.fromisoformat("2023-07-20")
hplc_corrections = acode.correctMetabolites(hplc_data,tracer_hl,tracer_name,date)
hdf = hplc_corrections
# Filter out rows that don't have "min" in their "Name" column
hdf = hdf[hdf['Name'].str.contains('min')]
# Remove rows with NaN values in the Percent_Intact column
hdf = hdf.dropna(subset=['Percent_Intact'])
hdf["times"] = hdf["Name"].apply(lambda x: int(x.split()[0]))
#sort hdf by times
hdf = hdf.sort_values(by=['times'])
pd.DataFrame(hdf.iloc[:, -1:-3:-1]).to_csv("C:/Users/rickr/Downloads/"+str(str(scan6.date)[:10]+'_'+tracer_name+'_Parent_Fraction.csv'), index = False)
pd.DataFrame(hplc_corrections).to_csv("C:/Users/rickr/Downloads/"+str(str(scan6.date)[:10]+'_'+tracer_name+'_Metabolite_Fractions.csv'), index = False)
