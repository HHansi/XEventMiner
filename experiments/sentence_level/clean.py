
with open("file.txt") as file_in:
    sentence_id = 0
    named_entity = []
    for line in file_in:
        if line.strip():

