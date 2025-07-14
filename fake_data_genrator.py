import random
from faker import Faker
import json

# Initialize Faker for random data generation
fake = Faker()

# Sample doctors, departments, and hospital locations
doctors = ["Dr. Smith", "Dr. Johnson", "Dr. Lee", "Dr. Brown", "Dr. Roshan", "Dr. Wilson", "Dr. Taylor", "Dr. Moore", "Dr. Anderson", "Dr. Thomas"]
departments = ["Dentist", "Cardiology", "Pediatrics", "Orthopedics", "Neurology", "Dermatology", "Radiology", "Psychiatry"]
hospital_locations = ["Patan Hospital", "City Hospital", "Global Care Hospital", "Sunshine Medical Center", "Mountain View Hospital"]

# Function to generate random appointment details for different users
def generate_appointment(user_name):
    doctor_name = random.choice(doctors)
    department = random.choice(departments)
    hospital_location = random.choice(hospital_locations)
    contact_number = fake.phone_number()
    appointment_date = fake.future_date(end_date="+1y").strftime("%Y-%m-%d")
    appointment_time = fake.time(pattern="%H:%M", end_datetime=None)
    
    # Keys in the format you requested
    appointment_doctor_key = f"{user_name}_doctor"
    appointment_date_key = f"{user_name}_appointment_date"
    appointment_department_key = f"{user_name}_checkup_department"
    appointment_hospital_key = f"{user_name}_hospital_location"
    appointment_contact_key = f"{user_name}_contact"
    appointment_time_key = f"{user_name}_appointment_time"
    
    # Appointment details
    appointment_data = {
        appointment_doctor_key: doctor_name,
        appointment_date_key: appointment_date,
        appointment_department_key: department,
        appointment_hospital_key: hospital_location,
        appointment_contact_key: contact_number,
        appointment_time_key: appointment_time
    }
    
    return appointment_data

# Generate 100 appointments for different users
appointments = {}
for i in range(100):
    user_name = fake.first_name()  # Random user name
    appointment = generate_appointment(user_name)
    
    # Merge into the main dictionary
    appointments.update(appointment)

# Save the generated data to a .json file
with open("appointments_data.json", "w") as json_file:
    json.dump(appointments, json_file, indent=4)

print("Data saved to 'appointments_data.json'")
