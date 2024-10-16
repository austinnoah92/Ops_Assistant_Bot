def preprocess_text(text):
    #simple preprocessing to remove extrace spaces and new line
    
    cleaned_text = " ".join(text.split())

    return cleaned_text