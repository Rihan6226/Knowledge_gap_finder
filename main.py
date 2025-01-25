import json
from utils import gap_finder,pretty_print




with open('LLQT.json', 'r') as file:
    data = json.load(file)
quiz_details = data['quiz']
with open('rJvd7g.json', 'r') as file:
    response_data = json.load(file)

result = gap_finder(quiz_details,response_data)
pretty_print(result)