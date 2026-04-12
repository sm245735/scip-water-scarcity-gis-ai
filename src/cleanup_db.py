#!/usr/bin/env python3
import psycopg2
conn = psycopg2.connect(
    host='db', port=5432, user='sm245735',
    password='DB_PASSWORD_PLACEHOLDER', database='thesis_analysis',
    connect_timeout=180
)
conn.set_isolation_level(0)  # autocommit
cur = conn.cursor()
try:
    cur.execute('TRUNCATE TABLE rainfall_grid_data')
    print('TRUNCATE 成功')
except Exception as e:
    print(f'TRUNCATE 失敗: {e}')
conn.close()
