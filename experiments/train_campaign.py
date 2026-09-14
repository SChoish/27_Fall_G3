"""Freeze and run actual source collection, quality-split training and evaluation.

Usage: python experiments/train_campaign.py --output /absolute/new/campaign
Use --resume only for the same frozen campaign. Per-job logs and hashes are saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]
QUALITIES=('low','medium','high','mixed')
LOCK=threading.Lock()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def initialize(root):
    root.mkdir(parents=True,exist_ok=False)
    for name in ('adc_driver','d2d_coupled_calibration'):
        shutil.copytree(REPO/name,root/name,ignore=shutil.ignore_patterns('results','data','checkpoints','tests','__pycache__'))
    for name in ('campaign_worker.py','campaign_report.py','train_campaign.py'):
        shutil.copy2(REPO/'experiments'/name,root/name)
    adc=root/'adc_driver';d2d=root/'d2d_coupled_calibration'
    sys.path.insert(0,str(d2d))
    import coupled_experiment as ce
    import coupled_link as cl
    cfg=ce.config()
    rng=__import__('numpy').random.default_rng(202609142347)
    contexts=[]
    for _ in range(68):
        common=dict(vdd=float(rng.uniform(.95,1.10)),temperature_c=float(rng.uniform(10,75)),rate_gbps=2.)
        lanes=[cl.link.Context(**common,resistance_ohm=float(rng.uniform(15,55)),
            inductance_nh=float(rng.uniform(.15,.5)),capacitance_pf=float(rng.uniform(.18,.4)),
            rx_capacitance_pf=float(rng.uniform(.04,.09)),clock_offset_ui=float(rng.uniform(-.2,.2))) for j in range(2)]
        contexts.append(asdict(cl.CoupledContext(*lanes,coupling_pf=float(rng.uniform(.02,.14)))))
    cfg.update(version='quality_campaign_20260914',contexts=contexts,train_ids=list(range(24)),
               validation_ids=list(range(24,36)),test_ids=list(range(36,68)),
               calibration_seed_base=902731,confirmation_seed_base=790211,
               episodes_per_context=12,checkpoint_steps=[5000,15000,50000],train_seeds=[0,1,2])
    write_json(d2d/'results/coupled_v1/config.json',cfg)
    protocol=dict(created_utc=utc(),profile='Three-seed, moderate offline acquisition budget',
        seeds=[0,1,2],qualities=list(QUALITIES),cpu_workers=8,cpu_affinity_cap=12,
        wall_time_cap_hours=6,minimum_free_disk_gib=6,
        adc=dict(source_designs_per_task=512,tasks=['nominal','robust'],total_source_designs=1024,
                 total_source_transients=2560,source_seeds={'nominal':2026091401,'robust':2026091402},
                 epochs=500,search_budget=128,training_fraction=.7,validation_fraction=.15,
                 quality_training_size=119,methods=['regression','rnc'],
                 main_attention='learned',main_line_search='none',
                 extra_mixed_rnc_ablations=[['off','none'],['off','armijo'],['learned','armijo']],
                 references='Existing three documented reference designs only; 25% of source perturbed locally, rest LHS. Warm-start search retains the common eight paid initial candidates.',
                 quality_definition='Training-only observed cost tertiles, equal-size groups; balanced mixed. Common validation/test.',
                 independent_sequences=[[.8,1.1,.8,1.1,.8,1.1],'uniform(.8,1.1), seed 271091','uniform(.8,1.1), seed 271092']),
        d2d=dict(train_contexts=24,validation_contexts=12,test_contexts=32,context_seed=202609142347,
                 pool_episodes=288,pool_transitions=9216,episodes_per_quality=96,transitions_per_quality=3072,
                 source_initial_measurements='One reference per episode and one extra measurement for non-reference starts; counted explicitly.',
                 methods=['bc','td3bc'],alphas=[.1,2.5],checkpoint_steps=[5000,15000,50000],
                 quality_definition='Within-context episode mean measured utility tertiles; mixed reuses balanced tiers. All types charged the entire shared acquisition pool.',
                 validation='Same 12 new contexts, all seeds jointly select checkpoint/alpha per quality; classical search receives the declared validation sweep.',
                 test='Same 32 unseen contexts and prespecified three independent packets; endpoints frozen before confirmation/oracle.'),
        cost_accounting='Report source pool acquisition, selected subset size, validation tuning and test measurements separately. Shared pools and mixed subsets are not multiplied by the number of datasets.',
        device_note='ngspice and the existing small MLP implementations execute on CPU; no GPU memory allocation is required.')
    write_json(root/'protocol.json',protocol)
    code=list(root.glob('*.py'))+list(adc.glob('*.py'))+list(d2d.glob('*.py'))
    code += [adc/'models/sky130_1v8_tt.spice',d2d/'assets/45nm_bulk.txt',d2d/'results/coupled_v1/config.json']
    write_json(root/'source_hashes.json',{str(p.relative_to(root)):digest(p) for p in code})
    write_json(root/'source_repository.json',dict(path=str(REPO),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)))


class Campaign:
    def __init__(self,root):
        self.root=root;self.started=time.monotonic();self.jobs={}
        self.worker=root/'campaign_worker.py';self.adc=root/'adc_driver';self.d2d=root/'d2d_coupled_calibration'
        self.env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                      NGSPICE_BIN=str(self.d2d/'runtime/usr/bin/ngspice'))
        self.state_path=root/'status.json'
        if self.state_path.exists():self.jobs=json.loads(self.state_path.read_text()).get('jobs',{})
        expected=json.loads((root/'source_hashes.json').read_text())
        for name,value in expected.items():
            if digest(root/name)!=value:raise ValueError('Frozen source changed: '+name)
        self.event('running')
    def event(self,state,name=None,record=None):
        with LOCK:
            if name:self.jobs[name]=record
            write_json(self.state_path,dict(state=state,updated_utc=utc(),elapsed_seconds=time.monotonic()-self.started,jobs=self.jobs))
            with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=utc(),state=state,job=name,record=record))+'\n')
    def run(self,name,folder,args,outputs,timeout=7200):
        expected=self.jobs.get(name,{})
        if expected.get('state')=='completed' and all(Path(p).exists() and digest(p)==expected.get('output_hashes',{}).get(str(Path(p).relative_to(self.root))) for p in outputs):
            return
        if shutil.disk_usage(self.root).free < 6*1024**3:raise RuntimeError('Less than 6 GiB free disk; stopping before another job.')
        log=self.root/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        command=[sys.executable,*map(str,args)]
        rec=dict(state='running',started_utc=utc(),command=command,cwd=str(folder),log=str(log.relative_to(self.root)))
        self.event('running',name,rec);print('START',name,flush=True)
        with log.open('a') as f:
            result=subprocess.run(command,cwd=folder,env=self.env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        rec.update(finished_utc=utc(),exit_code=result.returncode,state='completed' if result.returncode==0 else 'failed')
        if result.returncode==0:
            if not all(Path(p).is_file() for p in outputs):rec['state']='failed';rec['reason']='Missing output'
            else:rec['output_hashes']={str(Path(p).relative_to(self.root)):digest(p) for p in outputs}
        self.event('running',name,rec);print(rec['state'].upper(),name,flush=True)
        if rec['state']!='completed':raise RuntimeError(f'{name} failed; see {log}')
    def parallel(self,jobs,workers):
        with ThreadPoolExecutor(workers) as pool:
            futures=[pool.submit(self.run,*job) for job in jobs]
            for f in as_completed(futures):f.result()
    def worker_job(self,name,project,command,flags,outputs):
        return (name,project,[self.worker,command,'--project',project,*flags],outputs)
    def adc_pipeline(self):
        tasks={'nominal':None,'robust':self.adc/'configs/stress.json'}
        def contexts(path):return [] if path is None else ['--contexts',path]
        jobs=[]
        for j,(task,context) in enumerate(tasks.items()):
            data=self.adc/'data'/task/'pool.json'
            jobs.append(self.worker_job('adc_collect_'+task,self.adc,'adc_collect',
                ['--n',512,'--seed',2026091401+j,'--workers',2,'--output',data,*contexts(context)],[data]))
        self.parallel(jobs,2)
        for task in tasks:
            folder=self.adc/'data'/task
            self.run('adc_partition_'+task,self.adc,[self.worker,'adc_partition','--project',self.adc,'--input',folder/'pool.json','--output',folder],[folder/f'{q}.json' for q in QUALITIES])
        jobs=[]
        for task in tasks:
            for q in QUALITIES:
                for method in ('regression','rnc'):
                    for seed in (0,1,2):
                        checkpoint=self.adc/'checkpoints'/task/q/f'{method}_s{seed}.pt'
                        jobs.append((f'adc_train_{task}_{q}_{method}_s{seed}',self.adc,
                            ['run.py','train','--data',self.adc/'data'/task/f'{q}.json','--method',method,'--seed',seed,'--epochs',500,'--output',checkpoint],
                            [checkpoint,Path(str(checkpoint)+'.json')]))
        self.parallel(jobs,3)
        jobs=[];runs=[]
        for task,context in tasks.items():
            specs=[(q,m,'learned','none') for q in QUALITIES for m in ('regression','rnc')]
            specs += [('mixed','rnc',a,ls) for a,ls in [('off','none'),('off','armijo'),('learned','armijo')]]
            specs += [('none',m,'off','none') for m in ('random','ga')]
            for q,method,a,ls in specs:
                for seed in (0,1,2):
                    name=f'{task}_{q}_{method}_{a}_{ls}_s{seed}';folder=self.adc/'results/campaign'/name
                    flags=['--method',method,'--seed',seed,'--budget',128,'--attention',a,'--line-search',ls,'--output',folder,*contexts(context)]
                    if method in ('regression','rnc'):flags+=['--checkpoint',self.adc/'checkpoints'/task/q/f'{method}_s{seed}.pt']
                    jobs.append(self.worker_job('adc_search_'+name,self.adc,'adc_search',flags,[folder/'trace.json']))
                    runs.append(dict(name=name,task=task,quality=q,method=method,attention=a,line_search=ls,seed=seed,path=str(folder.relative_to(self.root))))
            for q in QUALITIES:
                name=f'{task}_{q}_lookup';folder=self.adc/'results/campaign'/name
                jobs.append(self.worker_job('adc_search_'+name,self.adc,'adc_lookup',
                    ['--input',self.adc/'data'/task/f'{q}.json','--output',folder,*contexts(context)],[folder/'trace.json']))
                runs.append(dict(name=name,task=task,quality=q,method='lookup',attention='off',line_search='none',seed=None,path=str(folder.relative_to(self.root))))
        write_json(self.root/'adc_runs.json',runs);self.parallel(jobs,4)
        jobs=[self.worker_job('adc_confirm_'+r['name'],self.adc,'adc_confirm',
              ['--input',self.root/r['path']/'trace.json','--output',self.root/r['path']],[self.root/r['path']/'confirmation.json']) for r in runs]
        self.parallel(jobs,4)
    def d2d_pipeline(self):
        out=self.d2d/'results/coupled_v1'
        for split in ('train','validation'):
            self.run('d2d_prepare_'+split,self.d2d,['coupled_experiment.py','prepare','--split',split,'--workers',4],[out/f'{split}_table.npz',out/f'{split}_table.npz.metadata.json'])
        self.run('d2d_collect_quality_pool',self.d2d,[self.worker,'d2d_collect','--project',self.d2d],
            [out/'dataset_pool.npz']+[out/'quality_datasets'/q/'dataset.npz' for q in QUALITIES])
        projects={}
        for q in QUALITIES:
            project=self.root/'d2d_qualities'/q
            if not project.exists():
                project.parent.mkdir(exist_ok=True)
                shutil.copytree(self.d2d,project,ignore=shutil.ignore_patterns('waves','quality_datasets','__pycache__'))
                target=project/'results/coupled_v1'
                for name in ('dataset.npz','dataset.npz.metadata.json','dataset.json'):
                    shutil.copy2(out/'quality_datasets'/q/name,target/name)
                (target/'waves').symlink_to(out/'waves',target_is_directory=True)
            projects[q]=project
            outputs=[project/'results/coupled_v1/models'/name/f'step{s}.pt' for s in (5000,15000,50000)
                for name in ([f'bc_s{i}' for i in (0,1,2)]+[f'td3bc_s{i}_a{a:g}' for a in (.1,2.5) for i in (0,1,2)])]
            self.run('d2d_train_'+q,project,['coupled_train.py','--all'],outputs)
            self.run('d2d_validate_'+q,project,['coupled_evaluate.py','validate'],[project/'results/coupled_v1/selection.json',project/'results/coupled_v1/validation.json'])
        self.run('d2d_prepare_test',self.d2d,['coupled_experiment.py','prepare','--split','test','--workers',4],[out/'test_table.npz',out/'test_table.npz.metadata.json'])
        for q,project in projects.items():
            target=project/'results/coupled_v1'
            for name in ('test_table.npz','test_table.npz.metadata.json'):shutil.copy2(out/name,target/name)
            self.run('d2d_calibrate_'+q,project,['coupled_evaluate.py','calibration'],[target/'test_calibration.json'])
        # All methods' settings are now frozen. Fill the shared independent cache
        # once, preventing concurrent writers in the stock waveform cache.
        self.run('d2d_independent_grid',self.d2d,[self.worker,'d2d_oracle','--project',self.d2d,'--workers',4],[out/'independent_oracle.json'])
        for q,project in projects.items():
            target=project/'results/coupled_v1'
            self.run('d2d_confirm_'+q,project,['coupled_evaluate.py','confirm','--workers',4],[target/'test.json'])
            self.run('d2d_data_audit_'+q,project,['coupled_audit.py','data_audit'],[target/'data_audit.json'])
            self.run('d2d_evaluation_audit_'+q,project,['coupled_audit.py','evaluation'],[target/'evaluation_audit.json'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=Path(a.output).resolve()
    if not a.resume:initialize(root)
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:12])
    campaign=Campaign(root)
    try:
        with ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(campaign.adc_pipeline),pool.submit(campaign.d2d_pipeline)]
            for job in as_completed(jobs):job.result()
        campaign.run('final_report',root,['campaign_report.py','--root',root],[root/'REPORT.md',root/'summary.json'])
        campaign.event('completed')
    except Exception as e:
        campaign.event('failed', 'pipeline_error', dict(state='failed',error=repr(e)))
        raise

if __name__=='__main__':main()
