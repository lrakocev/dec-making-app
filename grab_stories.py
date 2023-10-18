import psycopg2
import pandas as pd
import math

sql_qry = "SELECT DISTINCT(tasktypedone) FROM human_dec_making_table_utep ORDER BY tasktypedone"
data = []

conn = psycopg2.connect(database='live_database', host='10.10.21.128', user='postgres', port='5432', password='1234')
cursor = conn.cursor()
cursor.execute(sql_qry, data)

raw_data = cursor.fetchall()

cursor.close()
conn.close()

unique_tasks = set([row[0] for row in raw_data])

for i in unique_tasks:
    print(i)