import json
from  preproces_data import preprocess_data

def getkeyvaluedata(path):

    with open(path, 'r') as f:
        key_value_data = json.load(f)

    new_key_value_data = {}
    # preprocess 
    for key, value in key_value_data.items():
        new_key = preprocess_data(key)
        new_value = preprocess_data(value)
        new_key_value_data[new_key] = new_value

    all_keys = [ i for i in new_key_value_data.keys()]

    return all_keys,new_key_value_data


