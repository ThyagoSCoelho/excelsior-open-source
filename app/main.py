import os
import sys
import time
import pika
import json
from datetime import datetime
from excelsior import Excelsior


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


excelsior = Excelsior()
excelsior.initialize()


username = 'guest'
password = 'guest'
host = 'localhost'
port = '5672'
queue='excelsior_develompent'

# Configuração da conexão com o RabbitMQ (incluindo nome de usuário e senha)
credentials = pika.PlainCredentials(username, password)
parameters = pika.ConnectionParameters(host, heartbeat=30, credentials=credentials)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()
 
def callback(ch, method, properties, body):

    success = False
    try:        
        data = json.loads(body)
        contents = json.loads(data)
        print()
        # print(method.delivery_tag)
        print(' [INFO] Inicianlizando processamento da mensage para audio ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Processing...')
        
        # Start the clock
        start_time = time.time()
        success = excelsior.process(contents["uuid"], contents["text"])
        
        # Stop the clock and print the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if success: 
            print(f"[INFO] O processamento durou {elapsed_time:.2f} segundos.")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            print(' [SUCCESS] Menssagem processada corretamente ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Processed !')
        else:
            print(f"[INFO] O processamento durou {elapsed_time:.2f} segundos.")
            print(' [ERROR] Não foi possivel preccesar')
    except Exception as e:
        print(
            f"[ERROR] Unable to save the model on the machine. Error: {e}"
        )        

def create_consumer():
    
    print(' [*] Waiting for messages. To exit press CTRL+C')
    
    channel.basic_consume(queue, on_message_callback=callback)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    create_consumer()
