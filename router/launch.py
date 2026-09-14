"""Run the official Gateway with fixed vLLM DP ranks as logical workers."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import typer


def main(config: Path, host: str = '127.0.0.1', port: int = 30000, dry_run: bool = False) -> None:
    data=json.loads(config.read_text())
    workers=data['workers']
    assert workers and len({w['port'] for w in workers})==len(workers)
    assert all(w['rank'] in [0,1] for w in workers)
    assert (Path(data['tokenizer'])/'tokenizer.json').exists(), 'Use the existing local tokenizer directory'
    commands=[
        [sys.executable,str(Path(__file__).with_name('rank_proxy.py')),str(config.resolve())],
        [sys.executable,'-m','sglang_router.launch_router','--host',host,'--port',str(port),
         '--worker-urls',*[f'http://127.0.0.1:{w["port"]}' for w in workers],
         '--policy','cache_aware','--balance-abs-threshold','8','--balance-rel-threshold','1.5',
         '--max-payload-size','67108864','--max-concurrent-requests','256','--request-timeout-secs','3600',
         '--history-backend','none','--disable-retries','--tokenizer-path',str(Path(data['tokenizer']).resolve())]]
    if dry_run:
        print(json.dumps(commands,indent=2));return
    bypass=','.join([os.environ.get('NO_PROXY',''),'127.0.0.1','localhost',*(urlsplit(w['endpoint']).hostname for w in workers)])
    environment=os.environ|{'HF_HUB_OFFLINE':'1','NO_PROXY':bypass,'no_proxy':bypass}
    processes=[]
    try:
        for command in commands:
            processes.append(subprocess.Popen(command,env=environment))
        for _ in range(60):
            probe=subprocess.run(['curl','--noproxy','*','-sf','--max-time','2',f'http://127.0.0.1:{port}/workers'],capture_output=True,text=True)
            if probe.returncode==0:
                status=json.loads(probe.stdout)
                if status['total']==len(workers) and all(w['is_healthy'] for w in status['workers']):break
            assert all(p.poll() is None for p in processes),'Router or proxy exited during startup'
            time.sleep(.5)
        else:raise RuntimeError('Gateway workers did not become healthy')
        print(f'Gateway ready on port {port}; {len(workers)} DP workers',flush=True)
        while all(p.poll() is None for p in processes):time.sleep(1)
        raise RuntimeError('Gateway or rank proxy exited')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            for _ in range(100):
                if process.poll() is not None:break
                time.sleep(.1)
            if process.poll() is None:process.kill()
            process.wait()

if __name__=='__main__':typer.run(main)
