import re

def clean_ocr(text):
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\b([A-Za-z])\s(?=[A-Za-z]\b)', r'\1', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'INR\s*([\d,]+)\s*/\s*-', r'₹\1', text)
    text = re.sub(r'INR\s*([\d,]+)', r'₹\1', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()