import csv
import logging
import os

# NB config has a buncha global variables
import config

logger = logging.getLogger()

def main():
    base_file = "2025.csv"
    flattened_data = load_to_flattened(os.path.join("data", base_file))
    write_flattened_file(os.path.join("processed", base_file), flattened_data)


if __name__ == "__main__":
    main()
