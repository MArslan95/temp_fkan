"""Run the fixed study on two GPUs after pilot budgeting and a clean preflight."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .protocol import experiment_matrix, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--artifacts',default='revision_artifacts')
    parser.add_argument('--devices',default='0,1')
    parser.add_argument('--gpu-hours',type=float,help='Budget chosen after training-only pilots')
    parser.add_argument('--execute',action='store_true',help='Without this, only print the workload')
    args=parser.parse_args()
    devices=args.devices.split(',')
    if not devices or len(set(devices))!=len(devices) or any(not x.isdigit() for x in devices):
        parser.error('Provide unique nonnegative GPU indices, e.g. 0,1')
    rows=experiment_matrix()
    root=Path(args.artifacts)
    existing=[r for r in rows if (root/'runs'/r['run_id']).exists()]
    pending=[r for r in rows if not (root/'runs'/r['run_id']).exists()]
    print(json.dumps(dict(planned=len(rows),existing_protected=len(existing),pending=len(pending),
                          devices=devices,execute=args.execute,gpu_hours=args.gpu_hours)))
    if not args.execute:
        return
    if args.gpu_hours is None or not args.gpu_hours>0:
        parser.error('--execute requires a positive --gpu-hours budget fixed after pilots')
    root.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,'-m','revision.runner']
    common=['--data-root',args.data_root,'--artifacts',args.artifacts]
    subprocess.run(base+['preflight']+common,check=True)
    # Prepare each shared split serially; never race two split writers.
    seen=set()
    for config in pending:
        key=(config['dataset'],config['protocol'],config['split_seed'])
        if key not in seen:
            subprocess.run(base+['prepare','--run-id',config['run_id']]+common,check=True)
            seen.add(key)
    active={};finished=[];used=0.0;queue=iter(pending);exhausted=False
    logs=root/'logs';logs.mkdir(exist_ok=True)
    try:
        while active or not exhausted:
            now=time.monotonic()
            consumed=used+sum(now-job['start'] for job in active.values())
            if consumed/3600>=args.gpu_hours:
                raise RuntimeError('GPU-hour budget reached; running jobs terminated and retained as failed')
            for gpu in devices:
                if gpu in active or exhausted:
                    continue
                config=next(queue,None)
                if config is None:
                    exhausted=True;break
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
                logfile=(logs/(config['run_id']+'.log')).open('x',encoding='utf-8')
                process=subprocess.Popen(base+['train','--run-id',config['run_id']]+common,
                                         env=env,stdout=logfile,stderr=subprocess.STDOUT)
                active[gpu]=dict(process=process,log=logfile,start=time.monotonic(),config=config)
                print('Started',config['run_id'],'on GPU',gpu,flush=True)
            for gpu,job in list(active.items()):
                code=job['process'].poll()
                if code is None:
                    continue
                used+=time.monotonic()-job['start'];job['log'].close();del active[gpu]
                finished.append(dict(run_id=job['config']['run_id'],returncode=code))
                if code:
                    raise RuntimeError('Study stopped after failed run '+job['config']['run_id'])
                print('Completed',job['config']['run_id'],flush=True)
            if active:
                time.sleep(2)
    finally:
        interrupted=[]
        for job in active.values():
            if job['process'].poll() is not None:
                used+=time.monotonic()-job['start']
                job['log'].close()
                finished.append(dict(run_id=job['config']['run_id'],returncode=job['process'].returncode))
                continue
            job['process'].terminate()
            try:
                job['process'].wait(timeout=15)
            except subprocess.TimeoutExpired:
                job['process'].kill();job['process'].wait()
            job['log'].close()
            used+=time.monotonic()-job['start']
            config=job['config'];interrupted.append(config['run_id'])
            write_json(root/'runs'/config['run_id']/'status.json',dict(
                status='failed',config=config,error='Launcher interrupted: budget, sibling failure or user interruption'))
        write_json(root/('launch_'+str(time.time_ns())+'.json'),dict(
            completed=finished,interrupted=interrupted,gpu_hours_used=used/3600,
            gpu_hours_budget=args.gpu_hours,existing_runs_protected=[r['run_id'] for r in existing]))


if __name__=='__main__':
    main()
