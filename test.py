import ollama
import pandas as pd

# check if file is empty
if 
data = pd.read_csv("data.csv")

def parse_response(response):
    # check if the response is in the correct format
    if "<Start>" not in response or "<End>" not in response:
        return "Error: Response not in correct format"
    if response.count("<Start>") > 1 or response.count("<End>") > 1:
        return "Error: Response not in correct format"
    response = response.split("\n")
    name = response[0].split(":")[1]
    heat_pump = response[1].split(":")[1]
    room = response[2].split(":")[1]
    return name, heat_pump, room


prompt = "Your are a tool designed to extract information from text documents. Extract the follosing information: \n\n"
" Name of the person: (String)\n"
" If the person owns a heat punmp: (True/False)\n"
" the room in which the heat pump is : (Kitchen/Living Room/Other)\n"
" Format your response as follows: \n"
" <Start>Name: John Doe\n"
" Heat Pump: True\n"
" Room: Kitchen<End>\n\n"
" If the information is not contained in the text write NaN instead of the information. \n"
" Do not write anything else, just this response starting with the start tag and ending with the end tag\n"
" Here is the text:\n"


# load text dokument from file
text = open("text.txt", "r").read()

response = ollama.generate(prompt + text)

# parse response into the data df and add it to the data df
name, heat_pump, room = parse_response(response)

