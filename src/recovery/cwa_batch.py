"""Bounded gap-repair batch. Each station's failure is recorded; others proceed."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def main():
    recent=[f'2026-{m:02d}' for m in range(2,9)]
    jobs={
        'C0D580':['2018-10']+recent,
        '72D080':['2017-08','2018-03','2018-04','2019-05','2020-05','2020-08']+recent,
        'C0D550':recent,
        'C1D410':recent,
        'C1D420':recent,
        'C0D760':['2025-03','2025-09','2026-01']+recent,
        'C2D790':[f'2026-{m:02d}' for m in range(3,9)],
    }
    out=ROOT/'data/recovery/cwa_downloads'
    out.mkdir(parents=True,exist_ok=True)
    (out/'batch_plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    results={}
    with (out/'batch.log').open('a',encoding='utf-8') as log:
        for station,months in jobs.items():
            command=[sys.executable,str(ROOT/'src/recovery/cwa_fetch.py'),'--station',station,'--months',*months]
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=os.environ) as process:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                    log.flush()
                results[station]=process.wait()
            (out/'batch_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(results)

if __name__=='__main__':
    main()
