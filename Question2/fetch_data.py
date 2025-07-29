from datasets import load_dataset
import pandas as pd

data = load_dataset("antareepdey/Patient_doctor_chat")
print(data)

datasets = [i['Text'] for i in data['train']]

patient_query = []
doctor_response = []

for i in range(len(datasets)):
    inputs, outputs = datasets[i].split("###Output:")
    patient_query.append(inputs.replace("###Input:",""))
    doctor_response.append(outputs)

data = {"Patient query": patient_query, "Doctor response": doctor_response}
df = pd.DataFrame(data=data)
df.to_csv("data/patient_doctor.csv")