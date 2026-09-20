"""Checks of consequential parsing and recovery integrity, using the recovery DB."""
import sys
import unittest
from pathlib import Path
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/recovery'))
from recover import number

class RecoveryTests(unittest.TestCase):
    def test_negative_temperature_is_not_missing(self):
        self.assertEqual(number('-9.8','tx01'),(-9.8,'numeric'))
        self.assertEqual(number('-1.5','tx02'),(-1.5,'numeric'))
        self.assertIsNone(number('-9.8','pp01')[0])

    def test_quality_flags_are_not_zero_or_unflagged_numbers(self):
        for token in ('*0.0','*859.5','X','/','T','NaN','inf'):
            self.assertIsNone(number(token,'pp01')[0])
        self.assertEqual(number('0.0','pp01'),(0.0,'numeric'))
        self.assertEqual(number(None,'tx01'),(None,'not_provided'))

    def test_all_raw_rows_accounted_for(self):
        with psycopg.connect() as conn:
            self.assertEqual(conn.execute('''SELECT count(*) FROM recovery.sources s
                WHERE rows != (SELECT count(*) FROM recovery.raw_records r WHERE r.source_sha256=s.sha256)''').fetchone()[0],0)
            self.assertEqual(conn.execute('''SELECT count(*) FROM recovery.sources s WHERE kind LIKE 'weather_%'
                AND rows != (SELECT count(*) FROM recovery.weather w WHERE w.source_sha256=s.sha256)
                +(SELECT count(*) FROM recovery.parse_rejections r WHERE r.source_sha256=s.sha256)''').fetchone()[0],0)

    def test_research_identity_and_unique_days(self):
        with psycopg.connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM reservoirs WHERE reservoir_name='寶山第二水庫'").fetchone()[0],1)
            self.assertEqual(conn.execute('SELECT count(*) FROM recovery.stations WHERE stno=ANY(%s)',(['C0D580','C0D550','72D080','C1D410','C1D420','C0D760','C2D790'],)).fetchone()[0],7)
            self.assertEqual(conn.execute('SELECT count(*)-count(DISTINCT(stno,obs_date)) FROM recovery.weather_daily').fetchone()[0],0)

    def test_identical_files_keep_separate_months(self):
        with psycopg.connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM recovery.sources WHERE kind='weather_monthly'").fetchone()[0],859)
            self.assertEqual(conn.execute("SELECT count(DISTINCT extract(month FROM obs_date)) FROM recovery.weather WHERE stno='C0D760' AND obs_date<'2025-03-01'").fetchone()[0],2)

if __name__ == '__main__':
    unittest.main()
