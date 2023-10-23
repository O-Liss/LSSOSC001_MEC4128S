import logging
import matplotlib.pyplot as plt
import pandas as pd
import re

print("start")

#logging.basicConfig(level=logging.INFO, filename='example.log',filemode='w', format='%(asctime)s - %(levelname)s - %(message)s') # Filemode='w' overwrites the file each time
#X = 2
#logging.info("The value of X = %s", X)


# Configure logging to write to a log file
logging.basicConfig(
    level=logging.INFO,
    filename='../sea_ice_data.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Simulate logging sea ice data (replace with your actual data)
for frame_number in range(1, 11):
    sea_ice_concentration = 0.75 + frame_number * 0.05
    sea_ice_diameter = 5.0 + frame_number * 0.2
    ice_type = 'Type A'
    time = '2023-09-30 12:00:00'

    log_message = f"{time} {frame_number} - INFO - Frame {frame_number}: Concentration={sea_ice_concentration}"
    logging.info(log_message)

# Read the data from the log file using pandas
log_data = pd.read_csv('../sea_ice_data.log', header=None, names=['Timestamp', 'Level', 'Data'], delim_whitespace=True)

# Extract 'Frame Number' and 'Concentration' as integers and floats, handling missing values gracefully
log_data['Frame Number'] = pd.to_numeric(log_data['Data'].str.extract(r'Frame (\d+): Concentration=([\d.]+)').iloc[:, 0], errors='coerce')
log_data['Concentration'] = pd.to_numeric(log_data['Data'].str.extract(r'Frame (\d+): Concentration=([\d.]+)').iloc[:, 1], errors='coerce')

# Remove rows with missing data
log_data = log_data.dropna(subset=['Frame Number', 'Concentration'])

# Create the plot for Frame Number vs Sea Ice Concentration
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.plot(log_data['Frame Number'], log_data['Concentration'], marker='o', linestyle='-')
plt.xlabel('Frame Number')
plt.ylabel('Sea Ice Concentration')
plt.title('Frame Number vs Sea Ice Concentration')
plt.show()