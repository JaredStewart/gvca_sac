import csv
import logging
import os

import pandas as pd

# NB config has a buncha global variables
import config

logger = logging.getLogger()


def make_output_row(input_row):
    output_row = ["-"]*len(config.output_headers)
    for i, item in enumerate(input_row):
        if i < len(config.initial_headers):
            output_row[i] = input_row[i]
        elif item != '':
            input_header = config.input_headers[i]
            output_index = config.index_map[input_header]
            output_row[output_index] = item

    for field in config.additional_fields:
        output_row[config.index_map[field]] = config.additional_fields[field](output_row, config.index_map)

    return output_row


def load_to_flattened(filename):
    flattened_data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) != len(config.input_headers):
                logger.error("input_length not matched")
            if i < 2:
                continue
            flattened_data.append(make_output_row(row))
    
    return flattened_data


def write_flattened_file(filename, flattened_data):
    output_rows = []
    output_rows.append(config.output_headers)
    output_rows.extend(flattened_data)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)


def main():
    base_file = '2025.csv'
    flattened_data = load_to_flattened(os.path.join("data", base_file))
    write_flattened_file(os.path.join("processed", base_file), flattened_data)

if __name__ == "__main__":
    main()