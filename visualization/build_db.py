import pandas as pd
import sqlite3

import os
#print(os.getcwd())

file_path = 'visualization/distance_matrix_test.csv'
if os.path.exists(file_path):
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path)

    conn = sqlite3.connect('mydatabase.db')

    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS sample_dist_mat (
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        distance REAL,
        source_cluster INTEGER,
        CONSTRAINT pk_primary_key PRIMARY KEY (source, target)
    )
                ''')
    
    df.to_sql('sample_dist_mat', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Data loaded into database successfully.")
else:
    print("File not found:", file_path)