import os
import pandas as pd
import re

# Define the base path where the data folders are located
base_path = os.path.expanduser("~/Desktop/sproj'24-'25/ecmmodel(workingversion)/ECM_model/mergedmodel")

# Folder names
folder_names = ["lin_data", "rad_data", "uni_data"]

# Dictionary to hold displacement data for each folder
displacement_data = {"RunNumber": list(range(1, 21))}

# Create friendly gradient type names for the output columns
folder_to_column = {
    "lin_data": "Linear",
    "rad_data": "Radial",
    "uni_data": "Uniform"
}

# Process each folder
for folder in folder_names:
    folder_path = os.path.join(base_path, folder)
    mean_displacements = []
    
    try:
        # Get all .tsv files in the folder
        tsv_files = [f for f in os.listdir(folder_path) if f.endswith(".tsv")]
        
        # Sort files by run number if possible
        def extract_run_number(filename):
            match = re.search(r"run(\d+)", filename)
            if match:
                return int(match.group(1))
            return 0
        
        tsv_files.sort(key=extract_run_number)
        
        # Ensure we process exactly 20 files
        if len(tsv_files) < 20:
            print(f"Warning: Found only {len(tsv_files)} TSV files in {folder_path}. Expected 20.")
            # Pad with None for missing files
            mean_displacements = [None] * 20
        else:
            # Process first 20 files
            for i, filename in enumerate(tsv_files[:20]):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Read the TSV file
                    df = pd.read_csv(file_path, sep="\t")
                    
                    # Filter for rows with valid displacement values (N/A will be excluded)
                    valid_displacements = df['Displacement'].dropna()
                    
                    # Only include numeric values
                    valid_displacements = pd.to_numeric(valid_displacements, errors='coerce').dropna()
                    
                    # Calculate the mean displacement if we have valid values
                    if len(valid_displacements) > 0:
                        mean_disp = valid_displacements.mean()
                        mean_displacements.append(mean_disp)
                    else:
                        print(f"No valid displacement values found in {filename}")
                        mean_displacements.append(None)
                        
                except Exception as e:
                    print(f"Error processing {folder}/{filename}: {e}")
                    mean_displacements.append(None)
    
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        mean_displacements = [None] * 20
    
    # Add the mean displacements to our dictionary using the friendly name
    displacement_data[folder_to_column[folder]] = mean_displacements

# Create the summary DataFrame
result_df = pd.DataFrame(displacement_data)

# Save to TSV file in the base directory
output_file = os.path.join(base_path, "combined_mean_displacements.tsv")
result_df.to_csv(output_file, sep="\t", index=False)

print(f"Summary saved to: {output_file}")
print("\nPreview of the data:")
print(result_df.head(5))