from flask import Flask, jsonify, request, render_template
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('d3.html')

@app.route('/get_date_ranges')
def get_date_ranges():
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()
    cursor.execute("SELECT date_range FROM date_ranges ORDER BY date_range DESC")
    date_ranges = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(date_ranges)

@app.route('/get_stocks')
def get_stocks():
    cluster_id = request.args.get('cluster_id')
    end_date = request.args.get('date')  
    conn = sqlite3.connect('mydatabase.db')
    cur = conn.cursor()

    cur.execute('''
        SELECT ticker, residual_return, raw_return FROM stock_data 
        WHERE cluster = ? AND date = ?
        ''', (cluster_id, end_date))
    stocks = [{'ticker': row[0], 'residual_return': row[1], 'raw_return': row[2]} for row in cur.fetchall()]
    conn.close()
    return jsonify(stocks)

@app.route('/graph_data')
def graph_data():
    date_range = request.args.get('date_range')
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()
    
    if not date_range:
        date_range =  '19990511_to_19990810'
    
    nodes_table = f"graph_nodes_{date_range.replace('-', '')}"
    edges_table = f"graph_edges_{date_range.replace('-', '')}"

    cursor.execute(f"SELECT id, label, pos_x, pos_y FROM {nodes_table}")
    nodes = [{'id': f'Cluster {row[0]}', 'label': row[1], 'x': row[2], 'y': row[3]} for row in cursor.fetchall()]
    
    cursor.execute(f"SELECT source, target FROM {edges_table}")
    links = [{'source': row[0], 'target': row[1]} for row in cursor.fetchall()]

    conn.close()
    return jsonify({'nodes': nodes, 'links': links})


if __name__ == '__main__':
    app.run(debug=True)