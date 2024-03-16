import fireworks.client
from pydantic import BaseModel, ValidationError, Field
import json
from flask import request, jsonify, Blueprint
from langchain.output_parsers import PydanticOutputParser
import os

fireworks.client.api_key = os.environ.get('FIREWORKS_API_KEY')


class Category(BaseModel):
    category: str = Field(description="The category of the document submitted")


class Aadhar(BaseModel):
    name: str = Field(description="Name of the person the ID is issued to")
    dob: str = Field(description="Date of birth of the person as per the ID")
    gender: str = Field(description="Gender of the person as per the ID")
    aadharNumber: str = Field(description="12 - digit unique identification number as per the aadhar")
    VID: str = Field(description="16 digit VID number")
    issueDate : str = Field(description="Date of issue as per the aadhar")
    address: str = Field(description="Address as per the aadhar")

class DrivingLiscense(BaseModel):
    name: str = Field(description="Name as per the Driving License")
    dlNumber: str = Field(description="Driver's Liscense number (DL number)")
    DOI: str = Field(description="DOI (Date of issue)")
    validity: str = Field(description="Valid till date of the liscense")
    bloodGroup: str = Field(description="Blood group")
    address: str = Field(description="Address as per the Driving liscense")
    pincode: str = Field(description="Pincode/zipcode")

class PAN(BaseModel):
    name: str = Field(description="Name as per the PAN")
    PAN: str = Field(description="Permanent Account Number (PAN)")
    dob: str = Field(description="Date of birth of the person as per the ID")


def getPrompt(extracted_text, type):
    # aadhar_schema = Aadhar.model_json_schema()
    # pan_schema = PAN.model_json_schema()
    # DL_schema = DrivingLiscense.model_json_schema()

    if type == "aadhar card":
        prompt = f"""I will provide you with a Document called Aadhar, You have to extract all the named entities specified in the below schema and return the results in
        the same specified format

    ```json
    {{
        "name": "[Name of the person the ID is issued to]",
        "dob" : "[Date of birth of the person as per the ID]",
        "gender":"[Gender of the person as per the ID]",
        "aadharNumber": "[12 - digit unique identification number as per the aadhar]",
        "VID": "[16 digit VID number]",
        "issueDate" : "[Date of issue as per the aadhar]",
        "address" : "[Address as per the aadhar]"
    }}
    ```
    Make sure you only have the required json as output and no other text.

    {type} data: {extracted_text}

    The output should only have name, dob, gender, aadharNumber, VID, issueDate, address and no other additional information
    
    Important considerations:
    * only generate extracted data json.
    * Enclose property names in double quotes
    * The provided format should be strictly followed.
    * Do not generate any additional information other than the schema provided
    * If the value requested is missing, Default value based on datatype (eg: empty string for string type) must be provided

"""
        return prompt
    
    if type == "categorise":
        prompt = f"""
        You are provided with the text extracted from a document and you need to categorise its type based on its data as aadhar card, PAN card, DL (Driver's Liscence) any other document type based on the text.

        Here are some identifiers for the documemnts that will help you classify the documents:
        1) aadhar card: It is issued by Unique Identification Authority of India and contains 12 digit unique aadhar number and 16 digit VID number
        2) PAN card: It is issued by INCOME TAX DEPARTMENT GOVT. OF INDIA and contains 10 digit Permanent Account Number.
        3) DL : It contains DL number and authority to the user to drive certain class of vehicles

        Extracted Text: {extracted_text}

        You should strictly return your output in the following JSON format:
        ```json
        {{
            "catgory": "[The determined category]"
        }}
        ```

"""
        return prompt
    
    if type == "PAN card":
        prompt = f"""I will provide you with a Document called PAN card, You have to extract all the named entities specified in the below schema and return the results in
        the same specified format

    ```json
    {{
        "name": "[Name of the person the ID is issued to]",
        "PAN": "[Permanent Account Number (PAN)]",
        "dob" : "[Date of birth of the person as per the ID]"
    }}
    ```
    Make sure you only have the required json as output and no other text.

    {type} data: {extracted_text}

    The output should only have name, PAN, dob and no other additional information
    
    Important considerations:
    * only generate extracted data json.
    * Enclose property names in double quotes
    * The provided format should be strictly followed.
    * Do not generate any additional information other than the schema provided
    * If the value requested is missing, Default value based on datatype (eg: empty string for string type) must be provided

"""
        return prompt
    
    if type == "DL":
        prompt = f"""I will provide you with a Document called PAN card, You have to extract all the named entities specified in the below schema and return the results in
        the same specified format

    ```json
    {{
        "name": "[Name of the person the ID is issued to]",
        "dlNumber" : "[Driver's Liscense number (DL number)]",
        "DOI" : "[DOI (Date of issue)]",
        "validity": "[Valid till date of the license]",
        "bloodGroup": "[Blood Group]",
        "address" : "[Address as per the Driving liscense]",
        "pincode" : "[Pincode/zipcode]"
    }}
    ```
    Make sure you only have the required json as output and no other text.

    {type} data: {extracted_text}

    The output should only have name, dlNumber, DOI, validity, bloodGroup, address, pincode  and no other additional information
    
    Important considerations:
    * only generate extracted data json.
    * Enclose property names in double quotes
    * The provided format should be strictly followed.
    * Do not generate any additional information other than the schema provided
    * If the value requested is missing, Default value based on datatype (eg: empty string for string type) must be provided

"""
        return prompt
    


def handle_Escape(str):
    str = str.replace('"\\"','"').replace('\\"',"")
    return str


def askLLM(prompt):
    completion = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    messages=[
      {
        "role": "user",
        "content": prompt,
      }
    ],
    n=1,
    max_tokens=4000,
    temperature=0.1,
    top_p=0.9,
   )
    return completion.choices[0].message.content

parser_aadhar = PydanticOutputParser(pydantic_object=Aadhar)
parser_category = PydanticOutputParser(pydantic_object=Category)
pan_parser = PydanticOutputParser(pydantic_object=PAN)
DL_parser = PydanticOutputParser(pydantic_object=DrivingLiscense)


def categorise(text):
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        prompt = getPrompt(text, "categorise")
        completion = askLLM(prompt)
        test = handle_Escape(completion)
        try:
            results_values = parser_category.parse(test)
            return results_values.category
        except ValidationError as e:
            if attempt == max_attempts:
                return "Error"
            else:
                continue
        except:
            if attempt == max_attempts:
                return "Error"
            else:
                continue

def extract_data(text, category):
    if category == "aadhar card":
        parser = parser_aadhar
    elif category == "PAN card":
        parser = pan_parser
    elif category == "DL":
        parser = DL_parser
    else:
        return "Error"
    
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        prompt = getPrompt(text, category)
        completion = askLLM(prompt)
        # print(completion)
        test = handle_Escape(completion)
        try:
            results_values = parser.parse(test)
            return results_values
        except ValidationError as e:
            if attempt == max_attempts:
                return "Error"
            else:
                continue
        except:
            if attempt == max_attempts:
                return "Error"
            else:
                continue

main = Blueprint('main', __name__)

@main.route('/extract_invoice_text', methods=['POST'])
def extract_data_for_NER():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Text not found in request'}), 400
    text = data["text"]
    category = categorise(text)
    if category == "Error":
        return jsonify({"error":"Error finding a valid category"})
    
    res = extract_data(text,category)
    if res == "Error":
        return jsonify({"error":"Error extracting valid data"})
    
    return res.model_dump_json(), 200
        
