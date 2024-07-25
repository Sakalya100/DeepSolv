from dotenv import load_dotenv
import mysql.connector
import time
import os

load_dotenv()

def get_db_connection():
    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
        )
    cursor = connection.cursor()
    return connection, cursor

def create_application_logs():
    connection, cursor = get_db_connection()
    try:
        create_query = """
            CREATE TABLE application_logs(id INT AUTO_INCREMENT PRIMARY KEY, session_id varchar(255), user_query TEXT, gpt_response TEXT, model varchar(100),
            gpt_response_time varchar(100), input_token int, output_token int, total_token int, feedback bool, feedback_note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"""
        cursor.execute(create_query)
        connection.commit()
        print(f"Created application_logs table...")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        connection.close()

def insert_application_logs(session_id: str, user_query:str, gpt_response: str, model: str, gpt_response_time: str,
                            input_token: int, output_token: int, total_token: int, feedback: bool=None, feedback_note: str=None):
    connection, cursor = get_db_connection()
    try:
        insert_query = """
            INSERT INTO application_logs (session_id, user_query, response, model,
            response_time, input_tokens, output_tokens, total_tokens, feedback, feedback_note)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (session_id, user_query, gpt_response, model, gpt_response_time,
                  input_token, output_token, total_token, feedback, feedback_note)

        cursor.execute(insert_query, values)
        connection.commit()
        print("Record inserted into application_logs table...")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        connection.close()

def get_past_conversation(session_id):
    messages = []
    start_time = time.time()
    try:
        connection, cursor = get_db_connection()
        sql_query = "SELECT * from application_logs where session_id = %s;"
        cursor.execute(sql_query, (session_id, ))
        result = cursor.fetchall()
        for row in result:
            message_user = {"role": "human", "content": row[2]}
            message_assistant = {"role": "ai", "content": row[3]}
            messages.extend([message_user, message_assistant])
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    finally:
        connection.close()
        cursor.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"All session messages retrieved successfully, Time taken: {elapsed_time}")
    return messages



