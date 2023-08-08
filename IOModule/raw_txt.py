import os

def read_txt_file(raw_file_path, remove_tokens=["\n", " "], to_lower=True):
    with open(raw_file_path) as fr:
        data = fr.read()
    data = data.lower() if to_lower else data
    for rmv_token in remove_tokens:
        data = data.replace(rmv_token, "")
    return data