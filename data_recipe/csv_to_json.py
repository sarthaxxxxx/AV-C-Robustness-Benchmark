import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Create a list to store all rows
    data = []
    
    # Read CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create CSV reader object
        csv_reader = csv.DictReader(csv_file)
        
        # Convert each row into a dictionary and append to data list
        for row in csv_reader:
            data.append(row)
    
    output_dict = {'data': data}

    # Write JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        # Convert data to JSON and write to file
        json.dump(output_dict, json_file, indent=4)

    print(f"Conversion completed. JSON file saved as {json_file_path}")

# Example usage
if __name__ == "__main__":
    # Input and output file paths
    csv_file = "EPIC_100_validation.csv"  # Replace with your CSV file path
    json_file = "EPIC_100_validation.json"  # Replace with desired JSON output path
    
    # Convert CSV to JSON
    csv_to_json(csv_file, json_file)