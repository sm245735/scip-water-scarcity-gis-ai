"""Download explicit station-months through the public CODiS UI, retaining evidence."""
import argparse
import calendar
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/recovery/cwa_downloads'

def validate(path, month):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f)
        next(reader)
        columns=next(reader)
        if 'ObsTime' not in columns or 'Precp' not in columns:
            raise ValueError('Unrecognized English headers')
        rows=[]
        for row in reader:
            if not row:
                continue
            if len(row)!=len(columns):
                raise ValueError('CSV column count mismatch')
            rows.append(dict(zip(columns,row)))
    year,mon=map(int,month.split('-'))
    days=[]
    for row in rows:
        day=date(year,mon,int(row['ObsTime']))
        days.append(day.day)
        for key,value in row.items():
            if 'Time' in key and key!='ObsTime' and value and value[0:4].isdigit() and '/' in value:
                actual=value[:10].replace('/','-')
                midnight_next=(actual==(day+timedelta(days=1)).isoformat() and value[11:]=='00:00:00')
                if actual!=day.isoformat() and not midnight_next:
                    raise ValueError(f'CSV timestamp {actual} inconsistent with {day}')
    if len(days)!=len(set(days)):
        raise ValueError('Duplicate dates')
    return dict(status='no_observations' if not days else 'dates_complete' if len(days)==calendar.monthrange(year,mon)[1] else 'partial_dates',rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--station',required=True)
    p.add_argument('--start',help='YYYY-MM')
    p.add_argument('--end',help='YYYY-MM inclusive')
    p.add_argument('--months',nargs='+',help='Explicit non-contiguous YYYY-MM months')
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    legacy=ROOT.parent/'02. 程式/03. 政府資料/05. 中央氣象署/cwa_parser.py'
    spec=importlib.util.spec_from_file_location('legacy_cwa',legacy)
    cwa=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cwa)
    cwa.SAVE_DIR=str(OUT)
    cwa.LOG_FILE=str(OUT/'download.log')
    cwa.FAIL_LOG=str(OUT/'failure.log')
    names={v:k for k,v in cwa.STATIONS.items()}
    name=names[a.station]
    months=[]
    if a.months:
        months=sorted(set(a.months))
    else:
        if not a.start or not a.end:
            p.error('Provide --months or both --start and --end')
        y,m=map(int,a.start.split('-'))
        while f'{y:04d}-{m:02d}'<=a.end:
            months.append(f'{y:04d}-{m:02d}')
            y,m=(y+1,1) if m==12 else (y,m+1)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(accept_downloads=True,viewport={'width':1600,'height':1000})
        try:
            page.goto('https://codis.cwa.gov.tw/StationData',wait_until='domcontentloaded',timeout=60000)
            page.wait_for_timeout(10000)
            for kind in ['自動雨量站','自動氣象站','農業站']:
                page.get_by_role('checkbox',name=kind,exact=True).check()
            page.get_by_role('button',name='測站清單').click()
            page.wait_for_timeout(2000)
            if not cwa.setup_station(page,name,a.station):
                raise RuntimeError('Station selection failed')
            for month in months:
                target=OUT/f'{name}_{a.station}_{month.replace("-","")}.csv'
                status_path=target.with_suffix('.json')
                if target.exists() and status_path.exists():
                    old=json.loads(status_path.read_text())
                    if old.get('status')=='dates_complete' and old.get('sha256')==hashlib.sha256(target.read_bytes()).hexdigest():
                        print('Validated existing file:',target.name,flush=True)
                        continue
                result=dict(station=a.station,month=month,attempted_at=datetime.now(timezone.utc).isoformat())
                try:
                    if not cwa.select_month(page,*map(int,month.split('-'))):
                        raise RuntimeError('Month selection failed')
                    selected=page.evaluate(cwa.JS_GET_INPUT_VALUE)
                    if not selected or selected.replace('/','-')!=month:
                        raise RuntimeError(f'Wrong month in UI: {selected!r}')
                    page.wait_for_timeout(5000)
                    if not cwa.download_month_csv(page,name,a.station,month):
                        raise RuntimeError('Download failed')
                    result.update(validate(target,month))
                except Exception as exc:
                    result.update(status='failed',error=str(exc))
                status_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                print(result,flush=True)
                if result['status']=='failed':
                    raise RuntimeError(result['error'])
        finally:
            (OUT/f'{a.station}_last_page.html').write_text(page.content(),encoding='utf-8')
            page.screenshot(path=str(OUT/f'{a.station}_last_page.png'))
            browser.close()

if __name__=='__main__':
    main()
