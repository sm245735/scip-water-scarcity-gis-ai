"""Recover immutable CSV sources and report coverage in a dedicated database."""
import argparse
import calendar
import csv
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

csv.field_size_limit(64 * 1024 * 1024)  # WKT boundaries exceed the CSV default.

ROOT = Path('/app')
WORK = Path('/workspace')
FIELDS = ['pp01', 'tx01', 'tx02', 'rh01', 'wd01', 'ps01']
EN = ['Precp', 'Temperature', 'T Max', 'RH', 'WS', 'StnPres']
STATIONS = ['C0D580', 'C0D550', '72D080', 'C1D410', 'C1D420', 'C0D760', 'C2D790']
MONTH = re.compile(r'.+_([A-Z0-9]+)_(\d{6})\.csv$')
# Explicit station-month downloads validated against a companion JSON sidecar.
# They supersede both the legacy aggregate (priority 10) and the original monthly
# CSV bundle (priority 20) on a whole-row basis.
DOWNLOAD_DIR = ROOT / 'data/recovery/cwa_downloads'
DOWNLOAD_PRIORITY = 30
ALLOWED_DOWNLOAD_STATUS = ('dates_complete', 'partial_dates', 'no_observations')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def number(token, field):
    if token is None:
        return None, 'not_provided'
    token = token.strip()
    try:
        value = float(token)
    except ValueError:
        return None, 'missing' if not token else 'code:' + token
    if not math.isfinite(value):
        return None, 'nonfinite'
    if value in (-99, -99.9, -999, -9999, -999.9, -9999.9) or (value == -9.8 and field not in ('tx01', 'tx02')):
        return None, 'sentinel:' + token
    if (field not in ('tx01', 'tx02') and value < 0) or (field == 'rh01' and value > 100):
        return None, 'out_of_range'
    return value, 'numeric'

def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def load_source(conn, path, kind, monthly=False, source_priority=None, base=None):
    base = base or WORK
    content_sha = digest(path)
    relative = str(path.relative_to(base))
    sha = hashlib.sha256((relative + '\n' + content_sha).encode()).hexdigest()
    if conn.execute('SELECT 1 FROM recovery.sources WHERE sha256=%s', (sha,)).fetchone():
        return sha, False
    with path.open(encoding='utf-8-sig', newline='') as f:
        if monthly:
            reader = csv.reader(f)
            next(reader)
            header = next(reader)
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(f'CSV width mismatch: {path}')
                rows.append(dict(zip(header, row)))
        else:
            rows = list(csv.DictReader(f))
    if any(None in row for row in rows):
        raise ValueError(f'Unexpected CSV width: {path}')
    conn.execute('INSERT INTO recovery.sources(sha256,content_sha256,path,kind,bytes,rows) VALUES (%s,%s,%s,%s,%s,%s)',
                 (sha, content_sha, relative, kind, path.stat().st_size, len(rows)))
    with conn.cursor().copy('COPY recovery.raw_records FROM STDIN') as cp:
        for i, row in enumerate(rows, 1):
            cp.write_row((sha, i, Jsonb(row)))
    if kind in ('weather_legacy', 'weather_monthly', 'weather_download'):
        rejections = []
        match = MONTH.fullmatch(path.name) if monthly else None
        priority = source_priority if source_priority is not None else (20 if monthly else 10)
        with conn.cursor().copy('COPY recovery.weather FROM STDIN') as cp:
            for i, row in enumerate(rows, 1):
                if monthly:
                    stno, ym = match.groups()
                    try:
                        obs = date(int(ym[:4]), int(ym[4:]), int(row['ObsTime']))
                    except (ValueError, KeyError) as exc:
                        rejections.append((sha, i, f'invalid_date: {exc}'))
                        continue
                    tokens = [row.get(k) for k in EN]
                else:
                    stno = row['stno']
                    obs = date.fromisoformat(row['date'][:10])
                    tokens = [row.get(k.upper()) for k in FIELDS]
                vals, flags = [], {}
                for field, token in zip(FIELDS, tokens):
                    val, flag = number(token, field)
                    vals.append(val)
                    flags[field] = flag
                cp.write_row((sha, i, stno, obs, priority, *vals, Jsonb(flags)))
        if rejections:
            with conn.cursor() as cur:
                cur.executemany('INSERT INTO recovery.parse_rejections VALUES (%s,%s,%s)', rejections)
    return sha, True


def import_downloads(conn, source_dir=None):
    """Import paired station-month CSVs validated against their companion JSON.

    Each `<name>_<stno>_<YYYYMM>.csv` in `data/recovery/cwa_downloads/` must have
    a sibling `.json` whose `station`, `month`, `sha256` and `status` fields all
    agree with the file on disk; only `dates_complete`, `partial_dates`, and
    `no_observations` are accepted. Imported sources are tagged with the new
    `weather_download` kind and `source_priority=30`, so the existing `weather_daily`
    view keeps selecting the higher-priority download over the legacy (10) and
    original monthly (20) sources without mutating them.
    """
    source_dir = Path(source_dir) if source_dir else DOWNLOAD_DIR
    if not source_dir.exists():
        raise ValueError(f'Downloads directory missing: {source_dir}')
    pairs = []
    for csv_path in sorted(source_dir.glob('*.csv')):
        json_path = csv_path.with_suffix('.json')
        if not json_path.exists():
            raise ValueError(f'Missing companion JSON: {json_path}')
        info = json.loads(json_path.read_text())
        match = MONTH.fullmatch(csv_path.name)
        if not match:
            raise ValueError(f'Unrecognized filename: {csv_path.name}')
        stno, ym = match.groups()
        expected_month = f'{int(ym[:4]):04d}-{int(ym[4:]):02d}'
        if info.get('station') != stno:
            raise ValueError(f'Station mismatch in {json_path}: file={stno!r} json={info.get("station")!r}')
        if info.get('month') != expected_month:
            raise ValueError(f'Month mismatch in {json_path}: file={expected_month!r} json={info.get("month")!r}')
        content_sha = digest(csv_path)
        if info.get('sha256') != content_sha:
            raise ValueError(f'sha256 mismatch in {json_path}: json={info.get("sha256")!r} file={content_sha!r}')
        status = info.get('status')
        if status not in ALLOWED_DOWNLOAD_STATUS:
            raise ValueError(f'Disallowed status in {json_path}: {status!r}')
        pairs.append((csv_path, info, stno, ym))
    if not pairs:
        raise ValueError(f'No paired downloads found in {source_dir}')
    imported = []
    for i, (csv_path, info, stno, ym) in enumerate(pairs, 1):
        with conn.transaction():
            sha, new = load_source(conn, csv_path, 'weather_download', monthly=True,
                                    source_priority=DOWNLOAD_PRIORITY, base=ROOT)
        imported.append((csv_path, sha, new, info))
        if i % 20 == 0 or i == len(pairs):
            print(f'weather_download imports: {i}/{len(pairs)}', flush=True)
    new_count = sum(1 for _, _, n, _ in imported if n)
    print(f'weather_download imported: {new_count} new, {len(imported) - new_count} already present', flush=True)
    return imported

def restore_table(conn, sha, table, rename=None):
    rename = rename or {}
    columns = conn.execute('''SELECT column_name,data_type,udt_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position''', (table,)).fetchall()
    header = conn.execute('SELECT payload FROM recovery.raw_records WHERE source_sha256=%s LIMIT 1', (sha,)).fetchone()[0]
    expressions, names = [], []
    for name, datatype, udt in columns:
        key = rename.get(name, name)
        if key not in header:
            continue
        names.append(sql.Identifier(name))
        expr = sql.SQL("NULLIF(payload ->> {}, '')").format(sql.Literal(key))
        if udt == 'geometry':
            expr = sql.SQL('ST_SetSRID(ST_Force2D({}::geometry),4326)').format(expr)
            if name != 'geom' or table == 'reservoir_boundaries':
                expr = sql.SQL('ST_Multi({})').format(expr)
        else:
            expr = sql.SQL('{}::{}').format(expr, sql.SQL(datatype))
        expressions.append(expr)
    conn.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM recovery.raw_records WHERE source_sha256=%s ORDER BY row_number').format(
        sql.Identifier(table), sql.SQL(',').join(names), sql.SQL(',').join(expressions)), (sha,))
    if any(c[0] == 'id' for c in columns):
        conn.execute(sql.SQL("SELECT setval(pg_get_serial_sequence({},'id'), COALESCE(MAX(id),1), MAX(id) IS NOT NULL) FROM {}").format(sql.Literal(table), sql.Identifier(table)))

def restore(conn):
    with conn.transaction():
        conn.execute((ROOT / 'database/schema.sql').read_text())
        conn.execute((ROOT / 'database/recovery.sql').read_text())
    base = WORK / '03. 資料/09. 水庫蓄水量(日)'
    for name, table, rename in [
        ('reservoirs_202606141929.csv', 'reservoirs', {}),
        ('reservoir_daily_202605252126.csv', 'reservoir_daily', {}),
        ('reservoir_boundaries_202605252125.csv', 'reservoir_boundaries', {'reservoir_name': 'res_name'})]:
        with conn.transaction():
            sha, new = load_source(conn, base / name, table)
            if new:
                restore_table(conn, sha, table, rename)
        print(f'{table}: {"restored" if new else "unchanged; skipped"}', flush=True)
    weather = WORK / '03. 資料/08. 氣溫資料'
    with conn.transaction():
        sha, new = load_source(conn, weather / 'codis_st_202606280043.csv', 'stations')
        if new:
            conn.execute('''INSERT INTO recovery.stations SELECT payload->>'stno',payload->>'stname',
                NULLIF(payload->>'st_crtdt','')::date,NULLIF(payload->>'st_outdt','')::date,
                ST_SetSRID(ST_MakePoint((payload->>'longitude')::float,(payload->>'latitude')::float),4326),payload
                FROM recovery.raw_records WHERE source_sha256=%s''', (sha,))
    with conn.transaction():
        load_source(conn, weather / 'codis_weatherdata_202606202313.csv', 'weather_legacy')
    monthly = WORK / '02. 程式/03. 政府資料/05. 中央氣象署/CWA_Data'
    paths = sorted(p for p in monthly.glob('*.csv') if MONTH.fullmatch(p.name))
    if not paths:
        raise ValueError('No monthly files found')
    for i, path in enumerate(paths, 1):
        with conn.transaction():
            load_source(conn, path, 'weather_monthly', True)
        if i % 100 == 0 or i == len(paths):
            print(f'monthly files: {i}/{len(paths)}', flush=True)

def query_csv(conn, output, name, query, params=()):
    cur = conn.execute(query, params)
    columns = [c.name for c in cur.description]
    rows = [dict(zip(columns, row)) for row in cur]
    write_csv(output / name, rows, columns)
    return rows

def report(conn, output, end):
    output.mkdir(parents=True, exist_ok=True)
    sources = query_csv(conn, output, 'sources.csv', 'SELECT * FROM recovery.sources ORDER BY path')
    query_csv(conn, output, 'parse_rejections.csv', '''SELECT s.path,r.row_number,r.reason,p.payload
        FROM recovery.parse_rejections r JOIN recovery.sources s ON s.sha256=r.source_sha256
        JOIN recovery.raw_records p USING(source_sha256,row_number) ORDER BY s.path,r.row_number''')
    versions = conn.execute('SELECT version(),postgis_full_version()').fetchone()
    reservoirs = query_csv(conn, output, 'reservoir_coverage.csv', '''SELECT r.reservoir_id,r.reservoir_name,
        min(d.data_date) first_date,max(d.data_date) last_date,count(*) rows,
        count(*) FILTER(WHERE storage_rate IS NULL) null_storage,
        count(*) FILTER(WHERE observation_time::date IS DISTINCT FROM data_date) observation_date_mismatch,
        count(*) FILTER(WHERE storage_rate<0 OR storage_rate>100) storage_outside_0_100
        FROM reservoir_daily d JOIN reservoirs r USING(reservoir_id) GROUP BY 1,2 ORDER BY 1''')
    target = conn.execute("SELECT reservoir_id FROM reservoirs WHERE reservoir_name='寶山第二水庫'").fetchall()
    if len(target) != 1:
        raise ValueError(f'Target reservoir identity ambiguous: {target}')
    rid = target[0][0]
    query_csv(conn, output, 'bao2_daily.csv', '''SELECT d.*,
        coalesce(observation_time::date=data_date AND storage_rate BETWEEN 0 AND 100,false) AS strict_target_usable
        FROM reservoir_daily d WHERE reservoir_id=%s ORDER BY data_date''', (rid,))
    gaps = query_csv(conn, output, 'bao2_gaps.csv', '''SELECT day::date date,
        CASE WHEN d.id IS NULL THEN 'missing_row' WHEN storage_rate IS NULL THEN 'missing_storage'
        WHEN observation_time::date IS DISTINCT FROM data_date THEN 'observation_date_mismatch'
        ELSE 'storage_outside_0_100' END reason
        FROM generate_series('2014-01-01'::date,%s::date,'1 day') day
        LEFT JOIN reservoir_daily d ON d.data_date=day::date AND d.reservoir_id=%s
        WHERE d.id IS NULL OR storage_rate IS NULL OR observation_time::date IS DISTINCT FROM data_date
        OR storage_rate NOT BETWEEN 0 AND 100 ORDER BY day''', (end, rid))
    coverage = query_csv(conn, output, 'weather_yearly.csv', '''SELECT stno,extract(year FROM obs_date)::int AS year,
        min(obs_date) first_date,max(obs_date) last_date,count(*) rows,
        count(pp01) pp01_valid,count(tx01) tx01_valid,count(tx02) tx02_valid,
        count(rh01) rh01_valid,count(wd01) wd01_valid,count(ps01) ps01_valid
        FROM recovery.weather_daily GROUP BY 1,2 ORDER BY 1,2''')
    query_csv(conn, output, 'weather_quality.csv', '''SELECT stno,k.key variable,k.value flag,count(*) rows
        FROM recovery.weather w CROSS JOIN LATERAL jsonb_each_text(quality) k
        GROUP BY 1,2,3 ORDER BY 1,2,3''')
    query_csv(conn, output, 'weather_overlap.csv', '''SELECT stno,obs_date,
        count(*) source_rows,count(DISTINCT ROW(pp01,tx01,tx02,rh01,wd01,ps01)) distinct_value_sets
        FROM recovery.weather GROUP BY 1,2 HAVING count(*)>1 ORDER BY 1,2''')
    query_csv(conn, output, 'weather_source_duplicates.csv', '''SELECT source_sha256,stno,obs_date,count(*) rows
        FROM recovery.weather GROUP BY 1,2,3 HAVING count(*)>1''')
    query_csv(conn, output, 'identical_files.csv', '''SELECT content_sha256,count(*) files,
        max(rows) data_rows,string_agg(path,'; ' ORDER BY path) paths FROM recovery.sources
        GROUP BY content_sha256 HAVING count(*)>1''')
    query_csv(conn, output, 'research_weather_snapshot.csv', '''SELECT w.*,
        s.stname,s.opened,s.closed,
        (w.obs_date>=coalesce(s.opened,w.obs_date) AND w.obs_date<=coalesce(s.closed,w.obs_date)) within_station_history
        FROM recovery.weather_daily w JOIN recovery.stations s USING(stno)
        WHERE stno=ANY(%s) ORDER BY stno,obs_date''', (STATIONS,))
    query_csv(conn, output, 'research_station_summary.csv', '''SELECT s.stno,s.stname,s.opened,s.closed,
        min(w.obs_date) first_row,max(w.obs_date) last_row,count(w.obs_date) rows,
        max(w.obs_date) FILTER(WHERE pp01 IS NOT NULL AND w.obs_date>=s.opened
        AND (s.closed IS NULL OR w.obs_date<=s.closed)) last_numeric_precip,
        max(w.obs_date) FILTER(WHERE pp01 IS NOT NULL AND tx01 IS NOT NULL AND tx02 IS NOT NULL
        AND rh01 IS NOT NULL AND wd01 IS NOT NULL AND ps01 IS NOT NULL AND w.obs_date>=s.opened
        AND (s.closed IS NULL OR w.obs_date<=s.closed)) last_complete_six_variables
        FROM recovery.stations s LEFT JOIN recovery.weather_daily w USING(stno)
        WHERE s.stno=ANY(%s) GROUP BY s.stno ORDER BY s.stno''', (STATIONS,))
    query_csv(conn, output, 'geometry_validation.csv', '''SELECT reservoir_name,ST_SRID(geom) srid,
        ST_IsValid(geom) valid,ST_IsValidReason(geom) reason FROM reservoir_boundaries ORDER BY id''')
    months = query_csv(conn, output, 'monthly_files.csv', '''SELECT s.path,s.rows,
        min(w.obs_date) first_date,max(w.obs_date) last_date,count(DISTINCT w.obs_date) distinct_days,
        count(w.pp01) pp01_valid,count(w.tx01) tx01_valid,count(w.tx02) tx02_valid,
        count(w.rh01) rh01_valid,count(w.wd01) wd01_valid,count(w.ps01) ps01_valid
        FROM recovery.sources s LEFT JOIN recovery.weather w ON w.source_sha256=s.sha256
        WHERE s.kind='weather_monthly' GROUP BY s.sha256 ORDER BY s.path''')
    station_meta = {r[0]: r[1:] for r in conn.execute('SELECT stno,opened,closed FROM recovery.stations')}
    for month in months:
        station, ym = MONTH.fullmatch(Path(month['path']).name).groups()
        year, mon = int(ym[:4]), int(ym[4:])
        expected = calendar.monthrange(year, mon)[1]
        opened, closed = station_meta[station]
        first, last = date(year,mon,1), date(year,mon,expected)
        month['expected_calendar_days'] = expected
        month['status'] = ('before_station_opening' if opened and last < opened else
            'after_station_closure' if closed and first > closed else
            'no_observations' if month['distinct_days']==0 else
            'partial_dates' if month['distinct_days']!=expected else 'dates_complete')
        month['all_six_valid_days_not_asserted'] = True
    write_csv(output/'monthly_files.csv', months, list(months[0]))
    downloads = query_csv(conn, output, 'download_files.csv', '''SELECT s.path,s.bytes,s.rows,
        min(w.obs_date) first_date,max(w.obs_date) last_date,count(DISTINCT w.obs_date) distinct_days,
        count(w.pp01) pp01_valid,count(w.tx01) tx01_valid,count(w.tx02) tx02_valid,
        count(w.rh01) rh01_valid,count(w.wd01) wd01_valid,count(w.ps01) ps01_valid
        FROM recovery.sources s LEFT JOIN recovery.weather w ON w.source_sha256=s.sha256
        WHERE s.kind='weather_download' GROUP BY s.sha256 ORDER BY s.path''')
    if downloads:
        for download in downloads:
            station, ym = MONTH.fullmatch(Path(download['path']).name).groups()
            year, mon = int(ym[:4]), int(ym[4:])
            expected = calendar.monthrange(year, mon)[1]
            opened, closed = station_meta.get(station, (None, None))
            first, last = date(year, mon, 1), date(year, mon, expected)
            download['station'] = station
            download['month'] = ym
            download['expected_calendar_days'] = expected
            download['status'] = ('before_station_opening' if opened and last < opened else
                'after_station_closure' if closed and first > closed else
                'no_observations' if download['distinct_days'] == 0 else
                'partial_dates' if download['distinct_days'] != expected else 'dates_complete')
            download['all_six_valid_days_not_asserted'] = True
        write_csv(output/'download_files.csv', downloads, list(downloads[0]))
    observed = {}
    for stno, day, *vals in conn.execute('SELECT stno,obs_date,pp01,tx01,tx02,rh01,wd01,ps01 FROM recovery.weather_daily'):
        observed[stno, day] = vals
    weather_gaps, outside = [], []
    for stno in STATIONS:
        opened, closed = station_meta[stno]
        day = max(date(2014, 1, 1), opened or date(2014, 1, 1))
        stop = min(end, closed or end)
        fields = ['pp01'] if stno in ('C1D410', 'C1D420') else FIELDS
        while day <= stop:
            vals = observed.get((stno, day))
            missing = fields if vals is None else [k for k in fields if vals[FIELDS.index(k)] is None]
            if missing:
                weather_gaps.append(dict(stno=stno, date=day, reason='missing_row' if vals is None else 'missing_or_flagged', variables=';'.join(missing)))
            day += timedelta(days=1)
        for (station, day), vals in observed.items():
            if station == stno and ((opened and day < opened) or (closed and day > closed)):
                outside.append(dict(stno=stno, date=day, reason='outside_station_history'))
    write_csv(output/'weather_gaps.csv', weather_gaps, ['stno','date','reason','variables'])
    write_csv(output/'weather_outside_history.csv', outside, ['stno','date','reason'])
    ranges = []
    items = [dict(stno='bao2', date=g['date'], reason=g['reason']) for g in gaps] + weather_gaps
    for item in sorted(items, key=lambda r: (r['stno'], r['reason'], r.get('variables',''), r['date'])):
        key = (item['stno'], item['reason'], item.get('variables',''))
        if ranges and ranges[-1]['key'] == key and item['date'] == ranges[-1]['end'] + timedelta(days=1):
            ranges[-1]['end'] = item['date']
            ranges[-1]['days'] += 1
        else:
            ranges.append(dict(key=key, stno=key[0], reason=key[1], variables=key[2], start=item['date'], end=item['date'], days=1))
    write_csv(output/'backfill_ranges.csv', [{k:v for k,v in r.items() if k!='key'} for r in ranges], ['stno','reason','variables','start','end','days'])
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(), audit_end=str(end),
        database_versions=versions, sources=len(sources), raw_rows=sum(s['rows'] for s in sources),
        reservoir_rows=sum(r['rows'] for r in reservoirs), bao2=[r for r in reservoirs if r['reservoir_id']==rid][0],
        monthly_files=len(months), empty_monthly_files=sum(m['rows']==0 for m in months),
        download_files=len(downloads), empty_download_files=sum(d['rows']==0 for d in downloads),
        download_selected_rows=sum(d['distinct_days'] for d in downloads),
        weather_selected_rows=sum(r['rows'] for r in coverage), weather_gap_days=len(weather_gaps),
        bao2_gap_days=len(gaps), weather_outside_history=len(outside),
        policy='v1 conservative: flagged weather excluded; negative temperature retained except explicit sentinel codes; no imputation; download (priority 30) > monthly (20) > legacy (10) whole-row priority; reservoir date mismatch excluded from usable targets')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    manifest = {p.name:digest(p) for p in sorted(output.glob('*')) if p.is_file() and p.name != 'manifest.json'}
    manifest['code'] = {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__), ROOT/'database/schema.sql',ROOT/'database/recovery.sql']}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['restore','report','import-downloads'])
    p.add_argument('--end', type=date.fromisoformat, default=date(2026,9,18))
    p.add_argument('--output', type=Path, default=ROOT/'data/recovery/20260919')
    args = p.parse_args()
    with psycopg.connect(autocommit=True) as conn:
        if args.command == 'restore':
            restore(conn)
        elif args.command == 'import-downloads':
            import_downloads(conn)
        report(conn, args.output, args.end)

if __name__ == '__main__':
    main()
