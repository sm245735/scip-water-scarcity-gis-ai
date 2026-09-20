"""Host-side runner: durable command logs, checked exits, logical DB backup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ['docker', 'compose', '--env-file', '.env.recovery']

def run(command, log):
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        for line in iter(p.stdout.readline, b''):
            print(line.decode(errors='replace'), end='', flush=True)
            log.write(line)
            log.flush()
        code = p.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['restore','report','backup','test','verify-backup'])
    parser.add_argument('--end', default='2026-09-18')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs = ROOT/'logs/recovery'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs/f'{stamp}_{args.command}.log').open('xb') as log:
        if args.command in ('restore','report'):
            run(COMPOSE+['run','--rm','recovery',args.command,'--end',args.end], log)
        elif args.command == 'test':
            run(COMPOSE+['run','--rm','--entrypoint','python','recovery','-m','unittest','discover','-s','tests/recovery','-v'], log)
        elif args.command == 'verify-backup':
            target = sorted((ROOT/'data/recovery/backups').glob('*.dump'))[-1]
            dbname = 'recovery_check_' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
            run(COMPOSE+['exec','-T','db','createdb','-U','thesis',dbname], log)
            with target.open('rb') as f:
                subprocess.run(COMPOSE+['exec','-T','db','pg_restore','-U','thesis','--exit-on-error','--no-owner','-d',dbname],cwd=ROOT,stdin=f,stdout=log,stderr=log,check=True)
            query = 'SELECT (SELECT count(*) FROM reservoir_daily),(SELECT count(*) FROM recovery.raw_records),(SELECT count(*) FROM recovery.weather),(SELECT count(*) FROM recovery.sources);'
            counts = []
            for db in ('thesis_recovery',dbname):
                counts.append(subprocess.check_output(COMPOSE+['exec','-T','db','psql','-U','thesis','-d',db,'-At','-c',query],cwd=ROOT))
            if counts[0] != counts[1]:
                raise ValueError(f'Backup row counts differ: {counts}')
            log.write(b'Restored backup matches live table counts: '+counts[0])
            run(COMPOSE+['exec','-T','db','dropdb','-U','thesis',dbname], log)
            print('Backup restored successfully into a separate database; row counts verified.')
        else:
            backups = ROOT/'data/recovery/backups'
            backups.mkdir(parents=True, exist_ok=True)
            target = backups/f'thesis_recovery_{stamp}.dump'
            with target.open('xb') as f:
                subprocess.run(COMPOSE+['exec','-T','db','pg_dump','-U','thesis','-d','thesis_recovery','-Fc'], cwd=ROOT, stdout=f, stderr=log, check=True)
            # pg_restore is available in the DB image; inspect the host dump via stdin.
            with target.open('rb') as f:
                result = subprocess.run(COMPOSE+['exec','-T','db','pg_restore','--list'], cwd=ROOT, stdin=f, stdout=subprocess.PIPE, stderr=log, check=True)
            target.with_suffix('.list').write_bytes(result.stdout)
            h = hashlib.sha256()
            with target.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024), b''):
                    h.update(block)
            target.with_suffix('.json').write_text(json.dumps(dict(file=target.name,sha256=h.hexdigest(),bytes=target.stat().st_size),indent=2)+'\n')
            print(f'Backup complete; archive list verified: {target}')

if __name__ == '__main__':
    main()
