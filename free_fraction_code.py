import os
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import glob
import statistics
import xlrd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from openpyxl.drawing.image import Image


class FFScan:
    def __init__(self, sample_positions):
        self.tubes = ['A', 'B', 'C', 'D', 'E', 'F']
        self.sample_positions = sample_positions
        self.tracer = ""
        self.tracer_hl = None
        self.date = ""
        self.gc_sheet = None
        self.timing_sheet = None
        self.sample_count = {}
        self.bkg_value = None
        self.sample_values = {}
        self.tube_results = {}
        self.final_results = {}
        self.statistics_dict = {}
        self.df_final_results = None
        self.df_statistics = None
        self.df_combined = None
        self.std_dev = None
        self.saline_std = None
        self.average = None
        self.saline_average = None

    def get_input(self):
        self.tracer = str(input("Enter Radiotracer (F18):"))
        if "F18" in self.tracer:
            self.tracer_hl = 109.771
        elif "C11" in self.tracer:
            self.tracer_hl = 20.38

        date_input = str(input("Enter Date (Year.Month.Day):"))
        self.date = datetime.strptime(date_input, "%Y.%m.%d").strftime("%Y_%m_%d")

    def ff_data(self):
        self.gc_sheet = gc_sheet
        self.timing_sheet = timing_sheet

    def data_clean(self):
        gc_sheet = self.gc_sheet
        timing_sheet = self.timing_sheet

        print(timing_sheet.columns)

        gc_sheet['BARCODE'] = ""
        gc_sheet['START INJ'] = ""
        gc_sheet['FINISH INJ'] = ""

        sample_count = {}
        timing_sheet.columns = timing_sheet.columns.str.strip()

        for index, row in timing_sheet.iterrows():
            pos_blood = row['GC position blood']
            pos_plasma = row['GC position plasma']

            if pos_blood not in sample_count:
                sample_count[pos_blood] = 1
            else:
                sample_count[pos_blood] += 1
            gc_sheet.loc[gc_sheet['POS'] == pos_blood, 'BARCODE'] = f"WB {sample_count[pos_blood]}"
            gc_sheet.loc[gc_sheet['POS'] == pos_blood, 'START INJ'] = row['Draw Time start']
            gc_sheet.loc[gc_sheet['POS'] == pos_blood, 'FINISH INJ'] = row['Draw Time finish']

            if pos_plasma not in sample_count:
                sample_count[pos_plasma] = 1
            else:
                sample_count[pos_plasma] += 1
            gc_sheet.loc[gc_sheet['POS'] == pos_plasma, 'BARCODE'] = f"PL {sample_count[pos_plasma]}"
            gc_sheet.loc[gc_sheet['POS'] == pos_plasma, 'START INJ'] = row['Draw Time start']
            gc_sheet.loc[gc_sheet['POS'] == pos_plasma, 'FINISH INJ'] = row['Draw Time finish']

        gc_sheet.to_excel('updated_gamma_counter_sheet.xlsx', index=False)

    def calculate_bkg(self):
        gc_sheet = self.gc_sheet
        gc_sheet['BARCODE'].replace("", np.nan, inplace=True)
        background_rows = gc_sheet[gc_sheet['BARCODE'].isna()]
        bkg_value = np.mean(background_rows['CPM'])

        return bkg_value

    def parse_samples(self):
        sample_positions = self.sample_positions
        df = self.gc_sheet

        sample_values = {}

        for tube, positions in sample_positions.items():
            tube_sample_values = {}
            for position in positions:
                current_row = df[(df['POS'] == position) & (df['CH'] == 1)]
                next_row_empty = df['POS'].shift(-1).loc[current_row.index].isna().any()

                cpm1 = current_row['CPM'].values[0] if not current_row.empty else None
                eltime = current_row['ELTIME  '].values[0] if not current_row.empty else None

                if next_row_empty:
                    next_row_cpm2 = df['CPM'].shift(-1).loc[current_row.index].dropna()
                    if not next_row_cpm2.empty:
                        cpm2 = next_row_cpm2.values[0]

                tube_sample_values[position] = {
                    'CPM1': cpm1,
                    'CPM2': cpm2,
                    'ELTIME': eltime
                }

            sample_values[tube] = tube_sample_values
        return sample_values

    def column_f(self):
        # gc_sheet = self.gc_sheet
        # sample_values = self.sample_values
        # bkg_value = self.bkg_value

        tube_results = {}

        for tube, positions in self.sample_values.items():
            tube_results[tube] = {}
            for position, values in positions.items():
                cpm1 = values['CPM1']
                cpm2 = values['CPM2']
                eltime = values['ELTIME']

                calculation = (cpm1 + cpm2 - self.bkg_value) * 2 ** eltime / self.bkg_value
                tube_results[tube][position] = calculation

        self.tube_results = tube_results

        return tube_results, ff_scan

    def calculate_ff(self):
        sample_positions = self.sample_positions
        tube_results = self.tube_results

        final_results = {}

        for tube, positions in sample_positions.items():
            calculation_values = []
            for position in positions:
                if tube in tube_results and position in tube_results[tube]:
                    calculation_values.append(tube_results[tube][position])
                else:
                    calculation_values.append(0)  # if tube or position is missing, assume value of 0

            c_aliquot, c_remainder, c_top = calculation_values

            print("Tube:", tube, "Aliquot:", c_aliquot, "Remainder:", c_remainder, "Top:", c_top)

            if tube == list(sample_positions.keys())[0]:
                final_calculation = result = (c_aliquot / 0.15) / ((c_aliquot + c_remainder + c_top) / 0.5)

            else:
                final_calculation = result = (c_aliquot / 0.15) / ((c_aliquot + c_remainder + c_top) / 0.4)

            final_results[tube] = final_calculation

        self.final_results = final_results
        return final_results, ff_scan

    def calculate_stats(self):
        ff_values = list(self.final_results.values())[:4]
        saline = list(self.final_results.values())[-2:]

        average = statistics.mean(ff_values)
        std_dev = statistics.stdev(ff_values)
        saline_average = statistics.mean(saline)
        saline_std = statistics.stdev(saline)

        statistics_dict = {
            'Average Plasma': average,
            'Standard Deviation Plasma': std_dev,
            'Average Saline': saline_average,
            'Standard Deviation Saline': saline_std,
        }

        self.std_dev = std_dev
        self.saline_std = saline_std
        self.average = average
        self.saline_average = saline_average

        return statistics_dict, ff_scan, average, std_dev, saline_average, saline_std

    def save_results(self):
        # date = ff_scan.date
        date = datetime.strptime(self.date, "%Y_%m_%d")
        tracer = self.tracer

        save_directory = r'C:\Users\16203\Downloads\CNAMITestsRadData\CNAMI_Rad_Final_Results'

        df_final_results = pd.DataFrame(list(self.final_results.items()), columns=['Tubes', 'Values'])
        df_statistics = pd.DataFrame(list(self.statistics_dict.items()), columns=['Statistics', 'Values'])
        df_combined = pd.concat([df_final_results, df_statistics])

        data = {
            'Tubes': list(self.final_results.keys()),
            'Values': list(self.final_results.values()),
        }

        excel_filename = os.path.join(save_directory, f'FinalFF_{self.date}_{tracer}.xlsx')
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_combined.to_excel(writer, index=False, sheet_name=f'FF_{self.date}_{tracer}')

            fig, ax = plt.subplots()
            ax.scatter(range(len(df_final_results['Tubes'])), df_final_results['Values'], s=50)

            ax.set_xticks(range(len(df_final_results['Tubes'])))
            ax.set_xticklabels(df_final_results['Tubes'])
            ax.set_xlabel('Tube Labels', fontsize=14)
            ax.set_ylabel('Free Fraction', fontsize=14)
            ax.set_title(f'FF_{tracer}_{self.date}', fontsize=16)

            ax.text(0, data['Values'][0] + 0.03, 'Plasma', ha='center')
            ax.text(1, data['Values'][1] + 0.03, 'Plasma', ha='center')
            ax.text(2, data['Values'][2] + 0.03, 'Plasma', ha='center')
            ax.text(3, data['Values'][3] + 0.03, 'Plasma', ha='center')

            ax.errorbar(range(len(data['Tubes']) - 2), data['Values'][:-2], yerr=self.std_dev, fmt='none',
                        ecolor='blue', capsize=3, label='Plasma')
            ax.errorbar(range(len(data['Tubes']) - 2, len(data['Tubes'])), data['Values'][-2:],
                        yerr=self.saline_std, fmt='none', ecolor='purple', capsize=3, label='Controls')

            ax.legend()

            scatter_plot_filename = os.path.join(save_directory, f'{self.date}_{tracer}_scatter_plots.png')
            plt.savefig(scatter_plot_filename)

            scatter_plot_image = Image(scatter_plot_filename)
            workbook = writer.book
            worksheet_1 = writer.sheets[f'FF_{self.date}_{tracer}']
            worksheet_1.add_image(scatter_plot_image, 'F5')

        csv_filename = os.path.join(save_directory, f'{self.date}_{tracer}_FF.csv')
        df_final_results.to_csv(csv_filename, index=False)

        plt.show()

    def run(self):
        try:
            self.get_input()
            self.ff_data()
            self.data_clean()
            self.bkg_value = self.calculate_bkg()
            self.sample_values = self.parse_samples()
            self.column_f()
            self.calculate_ff()
            self.calculate_stats()
            self.save_results()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("An error occurred:", e)


if __name__ == "__main__":
    # Edit below #
    sample_positions = {
        'A': [150, 151, 152],
        'B': [154, 155, 156],
        'C': [158, 159, 160],
        'D': [162, 163, 164],
        'E': [166, 167, 168],
        'F': [170, 171, 172]
    }

    gc_sheet = pd.read_excel('/Users/16203/Downloads/CNAMITestsRadData/2023.06.16.AZAN.xls', skiprows=20)
    timing_sheet = pd.read_excel('/Users/16203/Downloads/CNAMITestsRadData/Timing Sheet GC pos 06162023 AZAN .xlsx',
                                 skiprows=5)

    ff_scan = FFScan(sample_positions)
    ff_scan.run()
