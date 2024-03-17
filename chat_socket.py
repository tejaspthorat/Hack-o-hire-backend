import socket
import threading
import fireworks.client
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
import fireworks.client
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
from redis.commands.search.query import Query
import requests
fireworks.client.api_key = "iaMTlSAgZeJnVAGOzKURAkw5Ux6NweLIrRxdBT80eycdcqXt"

url = "http://127.0.0.1:5000/process_data" 

HOST = '127.0.0.1'  
PORT = 8000 


def handle_Escape(str):
    str = str.replace('"\\"','"').replace('\\"',"")
    return str

def check_what_is_empty(user_peronal_details):
    ask_for = []
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0]:
            ask_for.append(f'{field}')
    return ask_for


def categorise_request(input_query):
    class Categorise(BaseModel):
        category: str = Field(...,
                              description="Categorise the user message",
                              enum = ["Emergency support", "General inquiries", "Technical support","Account management","Order status and tracking","Feedback and suggestions","Appointment scheduling"]
                              )
    prompt = f"""You are responsible to categorise the user question of banking related conversation in the following categories.

        Emergency support: Users might encounter urgent issues or emergencies that require immediate attention.
        General inquiries: Users might have general questions about your product, service, or company.
        Technical support: Users might encounter technical issues or bugs and need assistance troubleshooting or resolving them.
        Account management: Users might need help with tasks related to their account, such as password resets, account setup, or profile management.
        Order status and tracking: Users might want to inquire about the status of their orders or track their shipments.
        Feedback and suggestions: Users might want to provide feedback or suggestions for improving your products, services, or customer experience.
        Appointment scheduling: If required, users might want to schedule appointments or consultations with my company.


        user: {input_query}

        The output should be strictly in the following format
        The output should be striclty in json format and follow the below schema
        ```json
        {{
            "category": "[The category of user message]"
        }}
        ```
    """
    completion = askLLM(prompt)
    completion = handle_Escape(completion)
    parser = PydanticOutputParser(pydantic_object=Categorise)
    final_result = parser.parse(completion)
    return final_result


def schedule(input_query,client_socket):
    """
    Function to schedule an appointment

    """
    class ScheduleAppointment(BaseModel):
        time: Optional[str] = Field(...,
                                    description="The time for the appointment",
                                    )
        day: str = Field(description= "The day of the week when the apponintment should be scheduled")
        description: Optional[str] = Field(description= "Brief description of the nature of the meeting")

    ask_for = ["time", "day", "description"]
    
    user_Appointment_details = ScheduleAppointment(
                                time="",
                                day="",
                                description = ""
                                )
    
    def add_non_empty_details(current_details: ScheduleAppointment, new_details: ScheduleAppointment):
        non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details

    def getInfo(ask_for):
        prompt = f"""
            You are a smart AI assistant responsible to schedule appointments from the client in a conversational manner.
            You should only ask one question at a time even if you don't get all the info.
            Don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info.
            Here's what information you'll have to collect:
            1) time: Time of the appointment to be scheduled
            2) day: Day of the week when the appointment should be scheduled
            3) description: The nature of the meeting that the user wants to schedule

            You should only ask questions from the "askFor" list. Fields not in ask_for list have alreday been asked.
            If the ask_for list is empty, ask the user what else you can help him with.

            askFor: {ask_for}
        """
        completion = askLLM(prompt)

        return completion
    
    def filter_response(user_response):

        parser1 = PydanticOutputParser(pydantic_object=ScheduleAppointment)

        prompt = f"""
        You are a smart AI assistant responsible to extract the data provided in the below schema based on the provided user query

        ```json
        {{
            "time": "[Time when the user wants to schedule an appointment]",
            "day" : "[Day of whe week when the user wants to schedule an appointment]",
            "description": "[The nature of the meeting that the user wants to schedule]"
        }}
        ```
        User query: {user_response}

        only generate extracted data json.
        The provided format should be strictly followed.
        Populate empty string or default value corresponding to the datatype if the details are not found
        
        """
        completion = askLLM(prompt)
        completion = handle_Escape(completion)
        result_values = parser1.parse(completion)

        return result_values
    
    while len(ask_for)!=0 :
        client_socket.send(getInfo(ask_for).encode())
        user_input = client_socket.recv(1024).decode('utf-8')
        res = filter_response(user_input)
        res = add_non_empty_details(user_Appointment_details,res)
        user_Appointment_details = res
        print(user_Appointment_details)
        ask_for = check_what_is_empty(user_Appointment_details)

    client_socket.send("Appointment is scheduled, What else can i do for you?".encode())
    response = requests.post(url, json=user_Appointment_details.model_dump_json())
    print(response)
    return user_Appointment_details

def encode_query(user_query, embedder):
   encoded_queries = embedder.encode(user_query).astype(np.float32).tolist()
   return encoded_queries  

client = redis.Redis(host="redis-19988.c251.east-us-mz.azure.cloud.redislabs.com", port=19988, decode_responses=True, password = "OSzz3Wb89vOVsTceZZWyYPMXMVY1stgB")
embedder = SentenceTransformer("msmarco-distilbert-base-v4")

def RAG(user_input,client_socket):
    user_id = "Barclays"
    encoded_queries = encode_query(user_input,embedder)
    query = (
    Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'description')
     .dialect(2)
)
    similarity_search = client.ft(user_id).search(query,{"query_vector": np.array(encoded_queries, dtype=np.float32).tobytes()}).docs
    model_context = ""
    for i in similarity_search:
      model_context += i.description

    prompt = f"""You are a smart assistant built to help Barclays customers solve their banking related queries.
    You will be provided with context related to the user queries, you can use that context to build your answer. Be professional, and to the point with your answers.
    Try to answer in a short and consise way whithout overwhelming the user with information. Remember, you dont have to summarise the entire context but use only the useful part from it


    user_query: {user_input}

    context: {model_context}

    If the context provided does not answer user query, you can answer outside the context.
    
"""
    completion = askLLM(prompt)
    client_socket.send(completion.encode('utf-8'))

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


def handle_client(client_socket, addr):
    chat_history = []
    print(f"Accepted connection from {addr}")

    while True:
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break
        chat_history.append(data)
        print(f"Received message from {addr}: {data}")
        print(data)
        # client_socket.send(data.encode('utf-8'))
        # input_query = input(data)
        input_query = data
        category = categorise_request(input_query)
        print(category)
        if category.category == "Appointment scheduling":
            schedule("",client_socket)
        elif category.category == "General inquiries":
            RAG(input_query,client_socket)

    print(f"Connection from {addr} closed.")
    print(chat_history)
    client_socket.close()


def main():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          
    try:
        server_socket.bind((HOST, PORT))
        print(f"Running the server on {HOST} and port {PORT}")
    except:
        print(f"Unable to bind {HOST} and port {PORT}")

    server_socket.listen(5)

    print(f"Server listening on {HOST}:{PORT}")
    while True:
        client_socket, addr = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
        client_thread.start()


if __name__ == "__main__":
    main()