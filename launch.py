"""Launch the round9 TP4 x DP2 configuration using existing model weights."""
import json
import subprocess
from pathlib import Path

import typer


def main(model: Path, image: str = 'deepseek-v41-sm80:local', name: str = 'deepseek-v41-sm80',
         port: int = 8083, memory_utilization: float = 0.90, dry_run: bool = False) -> None:
    root=Path(__file__).resolve().parent
    if not (model/'config.json').is_file():
        raise ValueError('model must point to the existing DeepSeek-V4.1-Flash weights directory')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Use a lowercase Docker container name')
    if not 0 < memory_utilization < 1:
        raise ValueError('memory_utilization must be between 0 and 1')
    environment=json.loads((root/'config/environment.json').read_text())
    command=['docker','run','-d','--init','--name',name,'--gpus','all','--ipc=private','--shm-size','512g',
             '--ulimit','memlock=-1','-p',f'{port}:8083',
             '--mount',f'type=bind,src={model.resolve()},dst=/models/DeepSeek-V4.1-Flash,readonly',
             '--mount',f'type=volume,src={name}-runtime,dst=/workspace',
             '--mount',f'type=volume,src={name}-cache,dst=/root/.cache']
    for key,value in environment.items():
        command+=['--env',f'{key}={value}']
    serve=json.loads((root/'config/serve.json').read_text())
    serve[serve.index('--gpu-memory-utilization')+1]=str(memory_utilization)
    command+=[image,*serve[1:]]
    if dry_run:
        print(json.dumps(command,indent=2))
    else:
        subprocess.run(command,check=True)


if __name__=='__main__':
    typer.run(main)
