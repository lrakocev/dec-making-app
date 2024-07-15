import psycopg2
import pandas as pd
import math
import sys


if __name__ == "__main__":

    participant_id = sys.argv[1].rstrip()
    
    sql_qry = f"SELECT DISTINCT(tasktypedone) FROM human_dec_making_table_utep WHERE subjectidnumber = '{participant_id}' ORDER BY tasktypedone"
    data = []

    conn = psycopg2.connect(database='live_database', host='129.108.49.137', user='postgres', port='5432', password='1234')
    cursor = conn.cursor()
    cursor.execute(sql_qry, data)

    raw_data = cursor.fetchall()

    cursor.close()
    conn.close()

    unique_tasks = set([row[0] for row in raw_data])

    for i in unique_tasks:
        print(i)