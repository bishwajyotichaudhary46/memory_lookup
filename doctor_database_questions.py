import json

with open('appointments_data.json', 'r') as f:
    raw_data = json.load(f)


# Function to convert raw data into structured format
def transform_data(raw_data):
    transformed_data = []

    # Extracting each patient's data
    for key in raw_data:
        if "_doctor" in key:
            # Extracting the name of the person
            name = key.split("_")[0]
            doctor = raw_data.get(f"{name}_doctor")
            appointment_date = raw_data.get(f"{name}_appointment_date")
            department = raw_data.get(f"{name}_checkup_department")
            hospital_location = raw_data.get(f"{name}_hospital_location")
            contact = raw_data.get(f"{name}_contact")
            appointment_time = raw_data.get(f"{name}_appointment_time")

            # Creating a dictionary for each patient
            patient_data = {
                "name": name,
                "doctor": doctor,
                "appointment_time": appointment_time,
                "department": department,
                "hospital_location": hospital_location,
                "contact": contact
            }
            
            # Adding the structured data to the result
            transformed_data.append(patient_data)
    
    return transformed_data

# Transform the raw data into a list of dictionaries
formatted_data = transform_data(raw_data)

# Save the output as a new JSON file
with open('formatted_appointments.json', 'w') as output_file:
    json.dump(formatted_data, output_file, indent=4)

# Print the formatted data to verify
print(json.dumps(formatted_data, indent=4))