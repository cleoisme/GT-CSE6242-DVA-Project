from flask import Flask, render_template, jsonify, request
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('d3.html')

def get_data(table_name):
    conn = sqlite3.connect('mydatabase.db')
    query = f"SELECT source, target, distance, source_cluster FROM {table_name}"
    cursor = conn.cursor()
    cursor.execute(query)
    data = [{"source": row[0], "target": row[1], "distance": row[2], "source_cluster": row[3]} for row in cursor.fetchall()]
    conn.close()
    return data

@app.route('/data')
def data():
    table_name = request.args.get('table', 'sample_dist_mat')  
    results = get_data(table_name)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)