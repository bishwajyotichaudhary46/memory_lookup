import random
import json

# open json file in read mode and load into data
with open('formatted_appointments.json', 'r') as f:
    data = json.load(f)

# function to generate Q/A pairs for each user
def generate_qa(data):
    qa_pairs = []
    
    for entry in data:
        name = entry['name']
        doctor = entry['doctor']
        appointment_time = entry['appointment_time']
        department = entry['department']
        hospital_location = entry['hospital_location']
        contact = entry['contact']
        
        # Q/A Pair generation based on templates
        qa_pairs.append({
            "question": f"What time is {name}'s appointment?",
            "answer": f"The appointment time of {name} is {appointment_time}."
        })
        qa_pairs.append({
            "question": f"Who is {name}'s doctor?",
            "answer": f"The doctor for {name} is {doctor}."
        })
        qa_pairs.append({
            "question": f"Where is {name} having their appointment?",
            "answer": f"{name} is having their appointment at {hospital_location}."
        })
        qa_pairs.append({
            "question": f"What department is {name} visiting?",
            "answer": f"{name} is visiting the {department} department."
        })
        qa_pairs.append({
            "question": f"What is the contact number for {name}?",
            "answer": f"The contact number for {name} is {contact}."
        })
    
    # shuffle the Q/A pairs for more variability
    random.shuffle(qa_pairs)
    return qa_pairs

# generate the Q/A pairs
qa_pairs = generate_qa(data)

# extend data into 10 times
qa_pairs_extended = qa_pairs * 10  

# save to JSON file
with open('qa_pairs.json', 'w') as file:
    json.dump({"Q&A": qa_pairs_extended}, file, indent=4)

print(f"Generated {len(qa_pairs_extended)} Q/A pairs.")
