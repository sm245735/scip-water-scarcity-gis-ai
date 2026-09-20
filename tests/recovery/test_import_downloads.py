"""Verification of the new weather_download import pipeline.

Covers three guarantees called out in the spec:

* a re-run of `import_downloads` is idempotent: the source/raw/weather row counts
  do not grow, because `load_source` short-circuits on the existing identity hash;
* the raw row count reconciles with both `recovery.raw_records` and the
  `recovery.weather` + `recovery.parse_rejections` totals, including the new
  `weather_download` kind;
* the `weather_daily` view actually picks the higher-priority download row over
  any overlapping legacy (10) or monthly (20) row for the same (stno, obs_date).
"""
import sys
import unittest
from pathlib import Path
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/recovery'))


class ImportDownloadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = psycopg.connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def _count(self, query, params=()):
        return self.conn.execute(query, params).fetchone()[0]

    def test_weather_download_sources_present(self):
        count = self._count("SELECT count(*) FROM recovery.sources WHERE kind='weather_download'")
        self.assertGreater(count, 0, 'No weather_download sources were imported')

    def test_weather_download_priority_is_30(self):
        rows = self.conn.execute('''SELECT DISTINCT w.source_priority
            FROM recovery.weather w
            JOIN recovery.sources s ON s.sha256 = w.source_sha256
            WHERE s.kind = 'weather_download' ''').fetchall()
        self.assertGreater(len(rows), 0, 'No weather_download rows in recovery.weather')
        for (priority,) in rows:
            self.assertEqual(priority, 30, f'weather_download priority should be 30, got {priority}')

    def test_original_sources_untouched(self):
        # The 859 original monthly files and the legacy aggregate must not have
        # been mutated by import-downloads.
        self.assertEqual(self._count("SELECT count(*) FROM recovery.sources WHERE kind='weather_monthly'"), 859)
        self.assertEqual(self._count("SELECT count(*) FROM recovery.sources WHERE kind='weather_legacy'"), 1)
        self.assertEqual(self._count("SELECT count(DISTINCT source_priority) FROM recovery.weather WHERE source_priority IN (10, 20)"), 2)

    def test_idempotent_rerun_unchanged(self):
        from recover import import_downloads
        before_sources = self._count("SELECT count(*) FROM recovery.sources WHERE kind='weather_download'")
        before_raw = self._count('''SELECT count(*) FROM recovery.raw_records r
            JOIN recovery.sources s ON s.sha256 = r.source_sha256
            WHERE s.kind = 'weather_download' ''')
        before_weather = self._count('''SELECT count(*) FROM recovery.weather w
            JOIN recovery.sources s ON s.sha256 = w.source_sha256
            WHERE s.kind = 'weather_download' ''')
        with psycopg.connect(autocommit=True) as conn:
            import_downloads(conn)
        after_sources = self._count("SELECT count(*) FROM recovery.sources WHERE kind='weather_download'")
        after_raw = self._count('''SELECT count(*) FROM recovery.raw_records r
            JOIN recovery.sources s ON s.sha256 = r.source_sha256
            WHERE s.kind = 'weather_download' ''')
        after_weather = self._count('''SELECT count(*) FROM recovery.weather w
            JOIN recovery.sources s ON s.sha256 = w.source_sha256
            WHERE s.kind = 'weather_download' ''')
        self.assertEqual(before_sources, after_sources, 'sources count changed on idempotent re-run')
        self.assertEqual(before_raw, after_raw, 'raw_records count changed on idempotent re-run')
        self.assertEqual(before_weather, after_weather, 'weather count changed on idempotent re-run')

    def test_raw_row_reconciliation_for_downloads(self):
        # Every weather_download source must have all its rows present in raw_records.
        mismatches = self._count('''SELECT count(*) FROM recovery.sources s
            WHERE s.kind = 'weather_download'
              AND s.rows != (SELECT count(*) FROM recovery.raw_records r
                              WHERE r.source_sha256 = s.sha256) ''')
        self.assertEqual(mismatches, 0, 'weather_download sources with row-count mismatch in raw_records')
        # And weather_+rejections must equal the source's row count for every weather_* kind.
        like_pattern = 'weather_%'
        mismatches2 = self._count('''SELECT count(*) FROM recovery.sources s WHERE kind LIKE %s
            AND rows != (SELECT count(*) FROM recovery.weather w WHERE w.source_sha256 = s.sha256)
                     + (SELECT count(*) FROM recovery.parse_rejections r WHERE r.source_sha256 = s.sha256) ''',
            (like_pattern,))
        self.assertEqual(mismatches2, 0, 'weather_* sources with row-count mismatch in weather+rejections')

    def test_higher_priority_wins_in_daily_view(self):
        overlap = self.conn.execute('''SELECT w.stno, w.obs_date FROM recovery.weather w
            JOIN recovery.sources s ON s.sha256 = w.source_sha256
            WHERE s.kind = 'weather_download'
            INTERSECT
            SELECT w.stno, w.obs_date FROM recovery.weather w
            JOIN recovery.sources s ON s.sha256 = w.source_sha256
            WHERE s.kind = 'weather_monthly'
            ORDER BY 1, 2 LIMIT 5''').fetchall()
        self.assertGreater(len(overlap), 0,
                           'No overlapping (stno, obs_date) between downloads and monthly sources')
        for stno, obs_date in overlap:
            row = self.conn.execute('''SELECT s.kind, w.source_priority FROM recovery.weather_daily w
                JOIN recovery.sources s ON s.sha256 = w.source_sha256
                WHERE w.stno = %s AND w.obs_date = %s''', (stno, obs_date)).fetchone()
            self.assertIsNotNone(row, f'No row in weather_daily for {stno}/{obs_date}')
            kind, priority = row
            self.assertEqual(kind, 'weather_download',
                             f'On {stno}/{obs_date} expected download wins, got {kind} (priority {priority})')
            self.assertEqual(priority, 30,
                             f'On {stno}/{obs_date} expected priority 30, got {priority}')


if __name__ == '__main__':
    unittest.main()
