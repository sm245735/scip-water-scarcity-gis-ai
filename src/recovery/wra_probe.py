"""Probe the public replacement API using the same public client header as the site."""
import json
import re
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/recovery/source_probe'

def main():
    script=(OUT/'wra_app_readable.raw').read_text()
    key=re.search(r'o="([0-9a-f-]{36})"',script).group(1)
    session=requests.Session()
    session.headers.update({'apikey':key,'Referer':'https://fhy.wra.gov.tw/fhyv2/monitor/reservoir'})
    queries=[
        ('stations','https://fhy.wra.gov.tw/Api/v2/Reservoir/Station',{}),
        ('daily','https://fhy.wra.gov.tw/OpenApiv3/v2/Reservoir/Daily',{}),
        ('daily_2014','https://fhy.wra.gov.tw/OpenApiv3/v2/Reservoir/Daily',{'$filter':"StationNo eq '10405' and Time ge datetime'2014-01-01T00:00:00' and Time lt datetime'2014-01-03T00:00:00'"}),
        ('swagger','https://fhy.wra.gov.tw/OpenApiv3/swagger/v1/swagger.json',{}),
    ]
    for name,url,params in queries:
        try:
            r=session.get(url,params=params,timeout=(15,60))
            (OUT/f'wra_{name}.json').write_text(r.text,encoding='utf-8')
            print(name,r.status_code,len(r.content),r.text[:1000],flush=True)
        except requests.RequestException as exc:
            print(name,type(exc).__name__,str(exc),flush=True)
        time.sleep(1)

if __name__=='__main__':
    main()
