"""Inspect local XLS content and save bounded official-site connectivity probes."""
import hashlib
import io
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/recovery/source_probe'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for path in sorted((ROOT.parent/'03. 資料/09. 水庫蓄水量(日)').glob('*.xls')):
        raw = path.read_bytes()
        record = dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),signature=raw[:32].hex())
        try:
            if raw.startswith(bytes.fromhex('d0cf11e0a1b11ae1')):
                frames = [pd.read_excel(path, header=None)]
                record['format'] = 'Excel BIFF'
            else:
                frames = pd.read_html(io.StringIO(raw.decode('utf-8-sig')),flavor='lxml',header=None)
                record['format'] = 'HTML tables'
            record['tables'] = []
            for i, frame in enumerate(frames):
                target = OUT/f'{path.stem}_{i}.csv'
                frame.to_csv(target,index=False,encoding='utf-8-sig')
                bao = frame.loc[frame.iloc[:,0].astype(str).str.contains('寶山第二水庫')]
                record['tables'].append(dict(rows=len(frame),bao2=bao.astype(str).values.tolist(),output=str(target)))
        except Exception as exc:
            record['error'] = str(exc)
        inventory.append(record)
    (OUT/'local_xls.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(inventory,ensure_ascii=False,indent=2))
    results = []
    urls = {
        'wra_statistics':'https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx',
        'wra_chart':'https://fhy.wra.gov.tw/ReservoirPage_2011/ReservoirChart.aspx?key=10405',
        'codis':'https://codis.cwa.gov.tw/StationData',
        'wra_app':'https://fhy.wra.gov.tw/fhyv2/js/app.8496b7f3.js',
        'codis_app':'https://codis.cwa.gov.tw/Pages/StationData/StationData.js?06231715',
        'codis_global':'https://codis.cwa.gov.tw/StationData/global?06231715',
    }
    for name,url in urls.items():
        item = dict(name=name,url=url,attempted_at=datetime.now(timezone.utc).isoformat())
        try:
            r = requests.get(url,timeout=(15,45))
            (OUT/f'{name}.html').write_bytes(r.content)
            if name.endswith(('app','global')):
                snippets = [r.text[max(0,m.start()-160):m.end()+250] for m in re.finditer(r'(?i)(reservoir|GetReservoir|api/|baseURL|fetch\(|axios|Daily)',r.text)]
                (OUT/f'{name}_snippets.txt').write_text('\n\n'.join(snippets),encoding='utf-8')
            item.update(status=r.status_code,bytes=len(r.content),final_url=r.url,content_type=r.headers.get('Content-Type'))
        except requests.RequestException as exc:
            item['error'] = str(exc)
        print(item,flush=True)
        results.append(item)
    (OUT/'network.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()
