import os
import re

def preprocess_csv_units(input_path, output_path):
    """
    Cleans a plain CSV by removing [unit] from values and appending it to column names,
    based on the first occurrence in the data rows.
    Assumes the first line is the header and no lines start with '%'.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Extract header
    raw_headers = [h.strip() for h in lines[0].strip().split(',')]
    cleaned_headers = raw_headers.copy()

    # Process data
    output_lines = []
    output_lines.append(','.join(cleaned_headers) + '\n')  # temp, will update later

    for line in lines[1:]:
        values = [v.strip() for v in line.strip().split(',')]
        cleaned_values = []
        for i, val in enumerate(values):
            unit_match = re.search(r'(.*?)(\[.*?\])$', val)
            if unit_match:
                number, unit = unit_match.groups()
                cleaned_values.append(number)
                if '[' not in cleaned_headers[i]:  # only update once
                    cleaned_headers[i] = f"{raw_headers[i]} {unit}"
            else:
                cleaned_values.append(val)
        output_lines.append(','.join(cleaned_values) + '\n')

    # Replace the first line with updated headers
    output_lines[0] = ','.join(cleaned_headers) + '\n'

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    # input_file = os.path.join("data", "COMSOL", "results_3D_GE_Applied_Current_1MKOH_63_02_1MKOH_input_parameters_DOE_maximin_lhs_003.csv")
    # output_file = os.path.join("data", "COMSOL", "results_3D_GE_Applied_Current_1MKOH_63_02_1MKOH_input_parameters_DOE_maximin_lhs_processed_003.csv")
    # preprocess_csv_units(input_file, output_file)

    # import pandas as pd

    # log_path = "data/DNN_trained_models_docs.csv"

    # # Load existing log
    # df = pd.read_csv(log_path)

    # # If batch_norm column is missing, add it
    # if 'batch_norm' not in df.columns:
    #     df['batch_norm'] = False  # or "N/A" if you prefer

    # # Save updated log
    # df.to_csv(log_path, index=False)
    # print("Updated CSV to include 'batch_norm' column.")


    pass