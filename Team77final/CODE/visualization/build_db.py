import pandas as pd
import sqlite3
import networkx as nx
import os
from datetime import datetime, timedelta

#print(os.getcwd())

conn = sqlite3.connect('mydatabase.db')

cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        date DATE,
        ticker TEXT,
        residual_return REAL,
        cluster INTEGER,
        raw_return REAL,
        PRIMARY KEY (date, ticker)
    );''')

conn.commit()

def build_corr_mat(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    grouped = filtered_df.groupby(['date', 'cluster'])['residual_return'].mean()

    pivot_table = grouped.unstack(level='cluster')

    correlation_matrix = pivot_table.corr()
    return correlation_matrix



def build_graph(corr_mat):
    G = nx.Graph()

    for i in range(len(corr_mat)):
        for j in range(i + 1, len(corr_mat)):
            correlation = corr_mat.iloc[i, j]
            distance = ((1 - correlation) / 2) ** 0.5
            G.add_edge(f"Cluster {i}", f"Cluster {j}", weight=distance)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    pos = nx.spring_layout(G)
    return mst, pos


def save_graph_to_sqlite(mst, pos, conn, table_suffix):
    cursor = conn.cursor()
    
    nodes_table = f'graph_nodes_{table_suffix}'
    edges_table = f'graph_edges_{table_suffix}'

    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {nodes_table} (
        id INTEGER PRIMARY KEY,
        label TEXT,
        pos_x REAL,
        pos_y REAL
    );
    ''')
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {edges_table} (
        source INTEGER,
        target INTEGER,
        FOREIGN KEY(source) REFERENCES {nodes_table}(id),
        FOREIGN KEY(target) REFERENCES {nodes_table}(id)
    );
    ''')
    
    for node in mst.nodes():
        id = node.split(" ")[1]
        x, y = pos[node]
        cursor.execute(f"INSERT INTO {nodes_table} (id, label, pos_x, pos_y) VALUES (?, ?, ?, ?)", 
                       (id, node, x, y))
    
    for edge in mst.edges():
        cursor.execute(f"INSERT INTO {edges_table} (source, target) VALUES (?, ?)",
                       (edge[0], edge[1]))
    conn.commit()

#CREATE DATERANGE TABLE
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS date_ranges (
        id INTEGER PRIMARY KEY,
        date_range TEXT UNIQUE NOT NULL
    );
    ''')
conn.commit


def process_date_range(start_date, end_date, df, conn):
    table_suffix = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

    corr = build_corr_mat(df, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    
    mst, pos = build_graph(corr)

    save_graph_to_sqlite(mst, pos, conn, table_suffix)

#TEST THE FUNCTIONS
conn = sqlite3.connect('mydatabase.db')

corr = build_corr_mat(df,start_date='2017-10-23',end_date='2018-10-23')
mst,pos = build_graph(corr)

save_graph_to_sqlite(mst, pos, conn, 'sample')

conn.close()


conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

cursor.execute("SELECT MIN(date) FROM stock_data")
first_date = cursor.fetchone()[0]

cursor.execute("SELECT MAX(date) FROM stock_data")
last_date = cursor.fetchone()[0]


dates = pd.date_range(start=first_date, end=last_date, periods=100)

date_ranges = list(zip(dates[:-1], dates[1:]))

query = "SELECT * FROM stock_data"

df = pd.read_sql_query(query, conn)

for start, end in date_ranges:
    process_date_range(start, end, df, conn)
    print(f"processed range {start.date()} to {end.date()}")

conn.close()


def find_next_valid_date(conn, initial_date):
   
    current_date = initial_date
    while True:
        if check_data_availability(conn, current_date):
            return current_date
        current_date += timedelta(days=1)

def check_data_availability(conn, date):

    query = f"SELECT EXISTS(SELECT 1 FROM stock_data WHERE date='{date.strftime('%Y-%m-%d')}')"
    result = pd.read_sql_query(query, conn)
    return result.iloc[0, 0] == 1

def process_date_range(conn, start_date, end_date, df):

    adjusted_start_date = find_next_valid_date(conn, start_date)
    adjusted_end_date = find_next_valid_date(conn, end_date)

    #Add date range into db###############
    date_range = f"{adjusted_start_date.strftime('%Y%m%d')}_to_{adjusted_end_date.strftime('%Y%m%d')}"
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO date_ranges (date_range) VALUES (?)", (date_range,))
        conn.commit()
    except sqlite3.IntegrityError:
        print("Date range already exists in the database.")
    ########################

    table_suffix = f"{adjusted_start_date.strftime('%Y%m%d')}_to_{adjusted_end_date.strftime('%Y%m%d')}"

    corr = build_corr_mat(df, start_date=adjusted_start_date.strftime('%Y-%m-%d'), end_date=adjusted_end_date.strftime('%Y-%m-%d'))
    mst, pos = build_graph(corr)
    save_graph_to_sqlite(mst, pos, conn, table_suffix)


#POPULATE DB WITH NODES AND EDGES GRAPHS
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

cursor.execute("SELECT MIN(date), MAX(date) FROM stock_data")
first_date, last_date = map(pd.to_datetime, cursor.fetchone())

dates = pd.date_range(start=first_date, end=last_date, periods=100)
date_ranges = list(zip(dates[:-1], dates[1:]))

query = "SELECT * FROM stock_data"
df = pd.read_sql_query(query, conn)
count  = 0 
for start, end in date_ranges:
    count +=1
    process_date_range(conn, start, end, df)
    print(count)

conn.close()

