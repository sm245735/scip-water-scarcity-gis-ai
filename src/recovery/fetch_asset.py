"""Save an explicitly requested public web asset and HTTP metadata."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('url')
    p.add_argument('name')
    a=p.parse_args()
    out=Path(__file__).resolve().parents[2]/'data/recovery/source_probe'
    out.mkdir(parents=True,exist_ok=True)
    r=requests.get(a.url,timeout=(15,60))
    (out/(a.name+'.raw')).write_bytes(r.content)
    (out/(a.name+'.txt')).write_text(r.text.replace(';',';\n'),encoding='utf-8')
    meta=dict(url=a.url,final_url=r.url,status=r.status_code,bytes=len(r.content),time=datetime.now(timezone.utc).isoformat())
    (out/(a.name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)

if __name__=='__main__':
    main()
