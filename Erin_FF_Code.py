# Erin_FF_Code.py

# Import Modules#
import pandas as pd
import numpy as np
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image


class ff_scan:
    pass


def ff_data(gc_sheet, timing_sheet):
    scan1 = ff_scan()
    scan1.gc_sheet = gc_sheet
    scan1.timing_sheet = timing_sheet
    scan1.date = datetime.fromisoformat(input("Enter Date of Scan (YYYY-MM-DD):"))
    scan1.tracer = str(input("Enter Radiotracer (F-18):"))
    if "F-18" in scan1.tracer:
        scan1.tracer_hl = 109.771
    elif "C-11" in scan1.tracer:
        scan1.tracer_hl = 20.38
    return scan1


def data_clean(ff_scan):
    gc_sheet = ff_scan.gc_sheet
    timing_sheet = ff_scan.timing_sheet

    gc_sheet['BARCODE'] = ""
    gc_sheet['START INJ'] = ""
    gc_sheet['FINISH INJ'] = ""

    sample_count = {}
    timing_sheet.columns = timing_sheet.columns.str.strip()

    for index, row in timing_sheet.iterrows():
        pos_blood = row['GC position blood']
        pos_plasma = row['GC position plasma']
        pos_prc = row['GC position PRC']

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

        if pos_prc not in sample_count:
            sample_count[pos_prc] = 1
        else:
            sample_count[pos_prc] += 1
        gc_sheet.loc[gc_sheet['POS'] == pos_prc, 'BARCODE'] = f"PRC {sample_count[pos_prc]}"
        gc_sheet.loc[gc_sheet['POS'] == pos_prc, 'START INJ'] = row['Draw Time start']
        gc_sheet.loc[gc_sheet['POS'] == pos_prc, 'FINISH INJ'] = row['Draw Time finish']

    gc_sheet.to_excel('updated_gamma_counter_sheet.xlsx', index=False)

    return ff_scan


def calculate_bkg(ff_scan):
    gc_sheet = ff_scan.gc_sheet
    gc_sheet['BARCODE'].replace("", np.nan, inplace=True)
    background_rows = gc_sheet[gc_sheet['BARCODE'].isna()]
    bkg_value = np.mean(background_rows['CPM'])

    return bkg_value, ff_scan


def find_positions(ff_scan):
    gc_sheet = ff_scan.gc_sheet
    tubes = ['A', 'B', 'C', 'D', 'E', 'F']
    sample_positions = {}
    valid_input = False

    while not valid_input:
        try:
            sample_positions['A'] = list(map(int, input("Enter sample positions for A (space-separated): ").split()))
            sample_positions['B'] = list(map(int, input("Enter sample positions for B (space-separated): ").split()))
            sample_positions['C'] = list(map(int, input("Enter sample positions for C (space-separated): ").split()))
            sample_positions['D'] = list(map(int, input("Enter sample positions for D (space-separated): ").split()))
            sample_positions['E'] = list(map(int, input("Enter sample positions for E (space-separated): ").split()))
            sample_positions['F'] = list(map(int, input("Enter sample positions for F (space-separated): ").split()))
            valid_input = True
        except ValueError:
            print("Invalid input format. Please enter space-separated integers.")

    # hard code, sample_positions format
    #     sample_positions = {
    #         'A': [109, 110, 111],
    #         'B': [113, 114, 115],
    #         'C': [117, 118, 119],
    #         'D': [121, 122, 123],
    #         'E': [125, 126, 127],
    #         'F': [129, 130, 131]

    return sample_positions, tubes, ff_scan


def parse_samples(ff_scan):
    sample_positions = ff_scan.sample_positions
    gc_sheet = ff_scan.gc_sheet
    df = gc_sheet

    sample_values = {}

    for tube, positions in sample_positions.items():
        tube_sample_values = {}
        for position in positions:
            current_row = df[(df['POS'] == position) & (df['CH'] == 1)]
            next_row_empty = df['POS'].shift(-1).loc[current_row.index].isna().item()

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
    return sample_values, ff_scan


def column_f(ff_scan):
    gc_sheet = ff_scan.gc_sheet
    sample_values = ff_scan.sample_values
    bkg_value = ff_scan.bkg_value

    tube_results = {}

    for tube, positions in sample_values.items():
        tube_results[tube] = {}
        for position, values in positions.items():
            cpm1 = values['CPM1']
            cpm2 = values['CPM2']
            eltime = values['ELTIME']

            calculation = (cpm1 + cpm2 - bkg_value) * 2 ** eltime / bkg_value
            tube_results[tube][position] = calculation

    return tube_results, ff_scan


def calculate_ff(ff_scan):
    sample_positions = ff_scan.sample_positions
    tube_results = ff_scan.sample_positions

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
    return final_results, ff_scan


def calculate_stats(final_results, ff_scan):
    ff_values = list(final_results.values())[:4]
    saline = list(final_results.values())[-2:]

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

    return statistics_dict, ff_scan, average, std_dev, saline_average, saline_std


def save_results(final_results, statistics_dict, results_values, std_dev, saline_std, date, tracer):

    df_final_results = pd.DataFrame(list(final_results.items()), columns=['Tubes', 'Values'])
    df_statistics = pd.DataFrame(list(statistics_dict.items()), columns=['Statistics', 'Values'])
    df_combined = pd.concat([df_final_results, df_statistics])

    data = {
        'Tubes': list(final_results.keys()),
        'Values': results_values,
    }

    excel_filename = f'FinalFF_{date}_{tracer}.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_combined.to_excel(writer, index=False, sheet_name=f'FF_{date}_{tracer}')

        fig, ax = plt.subplots()
        ax.scatter(range(len(df_final_results['Tubes'])), df_final_results['Values'], s=50)

        ax.set_xticks(range(len(df_final_results['Tubes'])))
        ax.set_xticklabels(df_final_results['Tubes'])
        ax.set_xlabel('Tube Labels', fontsize=14)
        ax.set_ylabel('Free Fraction', fontsize=14)
        ax.set_title(f'FF_{tracer}_{date}', fontsize=16)

        ax.text(0, data['Values'][0] + 0.03, 'Plasma', ha='center')
        ax.text(1, data['Values'][1] + 0.03, 'Plasma', ha='center')
        ax.text(2, data['Values'][2] + 0.03, 'Plasma', ha='center')
        ax.text(3, data['Values'][3] + 0.03, 'Plasma', ha='center')

        ax.errorbar(range(len(data['Tubes']) - 2), data['Values'][:-2], yerr=std_dev, fmt='none',
                    ecolor='blue', capsize=3, label='Plasma')
        ax.errorbar(range(len(data['Tubes']) - 2, len(data['Tubes'])), data['Values'][-2:],
                    yerr=saline_std, fmt='none', ecolor='purple', capsize=3, label='Controls')

        ax.legend()

        scatter_plot_filename = f'{date}_{tracer}_scatter_plots.png'
        plt.savefig(scatter_plot_filename)

        scatter_plot_image = Image(scatter_plot_filename)
        workbook = writer.book
        worksheet_1 = writer.sheets[f'FF_{date}_{tracer}']
        worksheet_1.add_image(scatter_plot_image, 'F5')

    csv_filename = f'{date}_{tracer}_FF.csv'
    df_final_results.to_csv(csv_filename, index=False)

    plt.show()
