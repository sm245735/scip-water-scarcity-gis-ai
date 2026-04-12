import os
#!/usr/bin/env python3
import psycopg2
conn = psycopg2.connect(host='db', port=5432, user='sm245735', password=os.getenv('DB_PASSWORD'), database='thesis_analysis', connect_timeout=60)
cur = conn.cursor()
cur.execute('SELECT count(*) FROM rainfall_grid_data')
print(f'總列數: {cur.fetchone()[0]:,}')
cur.execute("SELECT area_name, count(*) FROM rainfall_grid_data GROUP BY area_name ORDER BY area_name")
for r in cur.fetchall():
    print(f'{r[0]}: {r[1]:,}')
conn.close()
