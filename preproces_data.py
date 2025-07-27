def preprocess_data(text):
    # preprocess
    text = text.lower()
    text = text.replace('?','')
    text = text.replace("'","")
    text = text.replace(","," ")
    text = text.replace("1)"," ")
    text = text.replace("2)"," ")
    text = text.replace("3)"," ")
    text = text.replace("4)"," ")
    text = text.replace("."," ")
    text = text.strip()
    return text