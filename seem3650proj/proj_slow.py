import xml.etree.ElementTree as ET
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the XML file
tree = ET.parse('data/rawSpeedVol-all.xml')
root = tree.getroot()

# Create a dictionary to store the detector IDs and their corresponding speeds
detector_speeds = {}

# Iterate over each <period> element
for period in root.findall('.//periods/period'):
    # Iterate over each <detector> element
    for detector in period.findall('.//detectors/detector'):
        # Get the detector ID
        detector_id = detector.find('detector_id').text
        
        # Iterate over each <lane> element with lane_id = 'Slow Lane'
        for lane in detector.findall(".//lanes/lane[lane_id='Slow Lane']"):
            # Get the speed value
            speed = lane.find('speed').text
            
            # Store the detector ID and speed in the dictionary
            detector_speeds[detector_id] = float(speed)

# Read the CSV file
csv_file = pd.read_csv('data/traffic_speed_volume_occ_info.csv')

# Create a dictionary to store the average speed for each district
district_avg_speeds = {}

# Iterate over each row in the CSV file
for index, row in csv_file.iterrows():
    # Get the detector ID
    detector_id = row['AID_ID_Number']
    
    # Check if the detector ID is in the dictionary
    if detector_id in detector_speeds:
        # Get the speed value
        speed = detector_speeds[detector_id]
        
        # Get the district
        district = row['District']
        
        # Normalize the district name
        district = re.sub(r'[&]', ' and ', district).lower()
        district = re.sub(r'\s+', ' ', district).strip()

        # Combine "central and western" and "central  and  western" into "central and western"
        if district == "central and western" or district == "central  and  western":
            district = "central and western"
        
        # Combine "central and western" and "central & western" into "central and western"
        if district in ["central and western", "central & western"]:
            district = "central and western"
        
        # Add the speed to the district average speed
        if district in district_avg_speeds:
            district_avg_speeds[district]['total_speed'] += speed
            district_avg_speeds[district]['count'] += 1
        else:
            district_avg_speeds[district] = {'total_speed': speed, 'count': 1}
        
districts = []
average_speeds = []

# Calculate the average speed for each district
for district, avg_speed in district_avg_speeds.items():
    average_speed = avg_speed['total_speed'] / avg_speed['count']
    print(f"District: {district}, Average Speed: {average_speed:.2f}")
    districts.append(district)
    average_speeds.append(average_speed)

plt.bar(districts, average_speeds)
plt.xlabel('District')
plt.ylabel('Average Speed')
plt.title('Average Speed by District')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.show()

# Load the xlsx file
table_110_02001_en = pd.read_excel('data/Table 110-02001_en.xlsx')

# Extract the population density and district information
population_density = table_110_02001_en.iloc[np.r_[5:9, 10:15, 16:25], 13].values
districts_xlsx = table_110_02001_en.iloc[np.r_[5:9, 10:15, 16:25], 1].values

# Normalize the district names
normalized_districts = []
for district in districts_xlsx:
    district = re.sub(r'[&]', ' and ', district).lower()
    district = re.sub(r'\s+', ' ', district).strip()
    normalized_districts.append(district)

# Create a dictionary to map district names to indices in population_density
district_indices = {district: i for i, district in enumerate(normalized_districts)}

ordered_population_density = []
# Order the population density for graph plotting
for district, avg_speed in zip(districts, average_speeds):
    idx = district_indices[district]
    population_density_value = population_density[idx]
    ordered_population_density.append(population_density_value)

# Calculate the correlation between population density and average speed
average_speeds_array = np.array(average_speeds)
population_density_array = np.array(ordered_population_density)

correlation_coefficient = np.corrcoef(average_speeds_array, population_density_array)[0, 1]
print("Correlation Coefficient:", correlation_coefficient)

# Plot the correlation
plt.scatter(ordered_population_density, average_speeds)

# Add labels for each plot
for i, (district, x, y) in enumerate(zip(districts, ordered_population_density, average_speeds)):
    plt.annotate(f"{district}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Population Density')
plt.ylabel('Average Speed')
plt.title('Correlation between Population Density and Average Speed')
plt.show()

with open("output.txt", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(districts)
    writer.writerow(average_speeds)
    writer.writerow(ordered_population_density)