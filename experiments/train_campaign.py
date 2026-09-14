"""Freeze and run actual source collection, quality-split training and evaluation.

Usage: python experiments/train_campaign.py --output /absolute/new/campaign
Use --resume only for the same frozen campaign. Per-job logs and hashes are saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import datetime
import hashlib
import importlib.metadata
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


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen_files(root):
    """Inventory executable source, circuit assets and every declared configuration."""
    paths=[root/name for name in ('campaign_worker.py','campaign_report.py','train_campaign.py','protocol.json')]
    for project in ('adc_driver','d2d_coupled_calibration'):
        base=root/project
        paths.extend(base.glob('*.py'))
        for name in ('configs','models','assets','runtime'):
            folder=base/name
            paths.extend(p for p in folder.rglob('*') if p.is_file())
    paths.append(root/'d2d_coupled_calibration/results/coupled_v1/config.json')
    return {str(p.relative_to(root)):digest(p) for p in sorted(set(paths))}


def resolve_ngspice(root,env):
    requested=env.get('NGSPICE_BIN')
    if requested:
        binary=shutil.which(os.path.expanduser(requested),path=env.get('PATH',os.defpath))
        if not binary:raise RuntimeError(f'NGSPICE_BIN is not an executable: {requested}')
    else:
        binary=shutil.which('ngspice',path=env.get('PATH',os.defpath))
        if not binary:
            bundled=root/'d2d_coupled_calibration/runtime/usr/bin/ngspice'
            binary=str(bundled) if bundled.is_file() and os.access(bundled,os.X_OK) else None
        if not binary:raise RuntimeError('ngspice is missing; install it or set NGSPICE_BIN to its executable')
    # Resolve before changing child cwd, including explicit relative paths or names on PATH.
    return str(Path(binary).resolve())


def input_hashes(paths):
    hashes={}
    for value in paths:
        path=Path(value).resolve()
        if not path.exists():raise ValueError(f'Missing job input: {path}')
        files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        hashes.update({str(p.resolve()):digest(p) for p in files})
    return hashes


def artifacts(*paths):
    """D2D data/models are consumed together with their provenance sidecars."""
    return [p for value in paths for p in (Path(value),Path(str(value)+'.metadata.json'))]


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
    write_json(root/'source_hashes.json',frozen_files(root))
    write_json(root/'source_repository.json',dict(path=str(REPO),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)))


class Campaign:
    def __init__(self,root):
        self.root=root=Path(root).resolve();self.started=time.monotonic();self.jobs={}
        self.worker=root/'campaign_worker.py';self.adc=root/'adc_driver';self.d2d=root/'d2d_coupled_calibration'
        self.env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        self.env['NGSPICE_BIN']=resolve_ngspice(root,self.env)
        self.state_path=root/'status.json'
        if self.state_path.exists():self.jobs=json.loads(self.state_path.read_text()).get('jobs',{})
        self.frozen=json.loads((root/'source_hashes.json').read_text())
        self.verify_frozen()
        if digest(__file__)!=self.frozen['train_campaign.py']:
            raise ValueError('Runner differs from the frozen campaign; use the matching runner or a new campaign')
        self.execution=self.execution_identity()
        environment_path=root/'execution_environment.json'
        if environment_path.exists():
            if json.loads(environment_path.read_text())!=self.execution:
                raise ValueError('Execution environment changed; create a new campaign')
        elif self.jobs:
            raise ValueError('Legacy resume has no execution identity; create a new campaign')
        else:write_json(environment_path,self.execution)
        # Check all prior evidence before either pipeline can start new work.
        for name,record in self.jobs.items():
            contract=record.get('contract')
            if not contract:raise ValueError(f'Legacy job has no input contract: {name}; create a new campaign')
            if self.contract(contract['command'],contract['cwd'],contract['outputs'],contract['inputs'])!=contract:
                raise ValueError(f'Job inputs or execution changed: {name}; create a new campaign')
            if record['state']=='completed':self.verify_outputs(name,record)
        self.event('running')
    def verify_frozen(self):
        actual=frozen_files(self.root)
        changed=sorted(name for name in actual.keys()|self.frozen.keys() if actual.get(name)!=self.frozen.get(name))
        if changed:raise ValueError('Frozen source/configuration changed or incomplete: '+', '.join(changed))
    def execution_identity(self):
        versions={}
        for name in ('numpy','scipy','torch','pymoo'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
        return dict(schema_version=1,python=str(Path(sys.executable).resolve()),python_version=sys.version,
            python_sha256=digest(sys.executable),ngspice=self.env['NGSPICE_BIN'],
            ngspice_sha256=digest(self.env['NGSPICE_BIN']),packages=versions,
            environment={k:self.env.get(k) for k in ('LD_LIBRARY_PATH','PYTHONPATH','SPICE_LIB_DIR',
                'SPICE_SCRIPTS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    def contract(self,command,folder,outputs,inputs):
        return dict(schema_version=1,command=list(command),cwd=str(Path(folder).resolve()),
            inputs=sorted(str(Path(p).resolve()) for p in inputs),input_hashes=input_hashes(inputs),
            outputs=sorted(str(Path(p).resolve()) for p in outputs),
            frozen_source_hashes=self.frozen,execution=self.execution)
    def verify_outputs(self,name,record):
        expected=record.get('output_hashes',{})
        for value in record['contract']['outputs']:
            path=Path(value)
            if not path.is_file() or digest(path)!=expected.get(str(path.relative_to(self.root))):
                raise ValueError(f'Completed output changed or missing: {name}: {path}; create a new campaign')
    def event(self,state,name=None,record=None):
        with LOCK:
            if name:self.jobs[name]=record
            write_json(self.state_path,dict(state=state,updated_utc=utc(),elapsed_seconds=time.monotonic()-self.started,jobs=self.jobs))
            with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=utc(),state=state,job=name,record=record))+'\n')
    def run(self,name,folder,args,outputs,inputs=(),timeout=7200):
        folder=Path(folder).resolve();outputs=[Path(p).resolve() for p in outputs]
        command=[sys.executable,*map(str,args)]
        self.verify_frozen()
        if self.execution_identity()!=self.execution:raise ValueError('Execution environment changed during campaign')
        contract=self.contract(command,folder,outputs,inputs)
        expected=self.jobs.get(name,{})
        if expected:
            if expected.get('contract')!=contract:
                raise ValueError(f'Job command, inputs or outputs changed: {name}; create a new campaign')
            if expected.get('state')=='completed':
                self.verify_outputs(name,expected)
                return
        if shutil.disk_usage(self.root).free < 6*1024**3:raise RuntimeError('Less than 6 GiB free disk; stopping before another job.')
        log=self.root/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        rec=dict(state='running',started_utc=utc(),command=command,cwd=str(folder),
                 log=str(log.relative_to(self.root)),contract=contract)
        self.event('running',name,rec);print('START',name,flush=True)
        try:
            with log.open('a') as f:
                result=subprocess.run(command,cwd=folder,env=self.env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            rec['exit_code']=result.returncode
            if result.returncode:raise RuntimeError(f'Child exited with status {result.returncode}')
            self.verify_frozen()
            if self.execution_identity()!=self.execution or self.contract(command,folder,outputs,inputs)!=contract:
                raise ValueError('Inputs or execution changed during job')
            if not all(p.is_file() for p in outputs):raise RuntimeError('Missing output')
            rec.update(state='completed',output_hashes={str(p.relative_to(self.root)):digest(p) for p in outputs})
        except Exception as e:
            rec.update(state='failed',reason=repr(e))
            raise RuntimeError(f'{name} failed; see {log}: {e}') from e
        finally:
            rec['finished_utc']=utc();self.event('running',name,rec);print(rec['state'].upper(),name,flush=True)
    def parallel(self,jobs,workers):
        with ThreadPoolExecutor(workers) as pool:
            futures=[pool.submit(self.run,*job) for job in jobs]
            for f in as_completed(futures):f.result()
    def worker_job(self,name,project,command,flags,outputs,inputs=()):
        return (name,project,[self.worker,command,'--project',project,*flags],outputs,inputs)
    def adc_pipeline(self):
        tasks={'nominal':None,'robust':self.adc/'configs/stress.json'}
        def contexts(path):return [] if path is None else ['--contexts',path]
        jobs=[]
        for j,(task,context) in enumerate(tasks.items()):
            data=self.adc/'data'/task/'pool.json'
            jobs.append(self.worker_job('adc_collect_'+task,self.adc,'adc_collect',
                ['--n',512,'--seed',2026091401+j,'--workers',2,'--output',data,*contexts(context)],[data],inputs=[] if context is None else [context]))
        self.parallel(jobs,2)
        for task in tasks:
            folder=self.adc/'data'/task
            self.run('adc_partition_'+task,self.adc,[self.worker,'adc_partition','--project',self.adc,'--input',folder/'pool.json','--output',folder],[folder/f'{q}.json' for q in QUALITIES],inputs=[folder/'pool.json'])
        jobs=[]
        for task in tasks:
            for q in QUALITIES:
                for method in ('regression','rnc'):
                    for seed in (0,1,2):
                        checkpoint=self.adc/'checkpoints'/task/q/f'{method}_s{seed}.pt'
                        jobs.append((f'adc_train_{task}_{q}_{method}_s{seed}',self.adc,
                            ['run.py','train','--data',self.adc/'data'/task/f'{q}.json','--method',method,'--seed',seed,'--epochs',500,'--output',checkpoint],
                            [checkpoint,Path(str(checkpoint)+'.json')],[self.adc/'data'/task/f'{q}.json']))
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
                    inputs=[] if context is None else [context]
                    if method in ('regression','rnc'):inputs.append(self.adc/'checkpoints'/task/q/f'{method}_s{seed}.pt')
                    jobs.append(self.worker_job('adc_search_'+name,self.adc,'adc_search',flags,[folder/'trace.json'],inputs=inputs))
                    runs.append(dict(name=name,task=task,quality=q,method=method,attention=a,line_search=ls,seed=seed,path=str(folder.relative_to(self.root))))
            for q in QUALITIES:
                name=f'{task}_{q}_lookup';folder=self.adc/'results/campaign'/name
                jobs.append(self.worker_job('adc_search_'+name,self.adc,'adc_lookup',
                    ['--input',self.adc/'data'/task/f'{q}.json','--output',folder,*contexts(context)],[folder/'trace.json'],inputs=[self.adc/'data'/task/f'{q}.json',*([context] if context else [])]))
                runs.append(dict(name=name,task=task,quality=q,method='lookup',attention='off',line_search='none',seed=None,path=str(folder.relative_to(self.root))))
        write_json(self.root/'adc_runs.json',runs);self.parallel(jobs,4)
        jobs=[self.worker_job('adc_confirm_'+r['name'],self.adc,'adc_confirm',
              ['--input',self.root/r['path']/'trace.json','--output',self.root/r['path']],[self.root/r['path']/'confirmation.json'],inputs=[self.root/r['path']/'trace.json']) for r in runs]
        self.parallel(jobs,4)
    def clone_inputs(self,project):
        inputs=[]
        for name,checksum in self.frozen.items():
            if not name.startswith('d2d_coupled_calibration/'):continue
            path=project/Path(name).relative_to('d2d_coupled_calibration')
            if not path.is_file() or digest(path)!=checksum:
                raise ValueError(f'Frozen D2D clone changed: {path}')
            inputs.append(path)
        return inputs
    def d2d_pipeline(self):
        out=self.d2d/'results/coupled_v1'
        for split in ('train','validation'):
            self.run('d2d_prepare_'+split,self.d2d,['coupled_experiment.py','prepare','--split',split,'--workers',4],
                     artifacts(out/f'{split}_table.npz')+[out/f'{split}_simulation.json'])
        quality_outputs=[p for q in QUALITIES for p in artifacts(out/'quality_datasets'/q/'dataset.npz')+[out/'quality_datasets'/q/'dataset.json']]
        self.run('d2d_collect_quality_pool',self.d2d,[self.worker,'d2d_collect','--project',self.d2d],
            artifacts(out/'dataset_pool.npz')+quality_outputs,inputs=artifacts(out/'train_table.npz'))
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
            projects[q]=project;target=project/'results/coupled_v1'
            source_inputs=self.clone_inputs(project)
            data_inputs=artifacts(target/'dataset.npz',target/'train_table.npz')+[target/'dataset.json']
            outputs=[target/'models'/name/f'step{step}.pt' for step in (5000,15000,50000)
                for name in ([f'bc_s{i}' for i in (0,1,2)]+[f'td3bc_s{i}_a{alpha:g}' for alpha in (.1,2.5) for i in (0,1,2)])]
            self.run('d2d_train_'+q,project,['coupled_train.py','--all'],artifacts(*outputs),inputs=source_inputs+data_inputs)
            self.run('d2d_validate_'+q,project,['coupled_evaluate.py','validate'],[target/'selection.json',target/'validation.json'],
                     inputs=source_inputs+data_inputs+artifacts(target/'validation_table.npz')+[target/'models'])
        self.run('d2d_prepare_test',self.d2d,['coupled_experiment.py','prepare','--split','test','--workers',4],
                 artifacts(out/'test_table.npz')+[out/'test_simulation.json'])
        for q,project in projects.items():
            target=project/'results/coupled_v1'
            for name in ('test_table.npz','test_table.npz.metadata.json'):shutil.copy2(out/name,target/name)
            self.run('d2d_calibrate_'+q,project,['coupled_evaluate.py','calibration'],[target/'test_calibration.json'],
                inputs=self.clone_inputs(project)+artifacts(target/'test_table.npz',target/'dataset.npz')+[target/'selection.json',target/'models'])
        # All methods' settings are now frozen. Fill the shared independent cache
        # once, preventing concurrent writers in the stock waveform cache.
        self.run('d2d_independent_grid',self.d2d,[self.worker,'d2d_oracle','--project',self.d2d,'--workers',4],
            [out/'independent_oracle.json'],inputs=[p/'results/coupled_v1/test_calibration.json' for p in projects.values()])
        for q,project in projects.items():
            target=project/'results/coupled_v1';source_inputs=self.clone_inputs(project)
            self.run('d2d_confirm_'+q,project,['coupled_evaluate.py','confirm','--workers',4],
                [target/'test.json',target/'confirmation_tables.npz'],
                inputs=source_inputs+[target/'test_calibration.json']+[out/'waves'/pattern for pattern in json.loads((out/'config.json').read_text())['confirmation_patterns']])
            self.run('d2d_data_audit_'+q,project,['coupled_audit.py','data_audit'],[target/'data_audit.json'],
                inputs=source_inputs+artifacts(target/'dataset.npz',target/'train_table.npz')+[out/'waves/calibration'])
            self.run('d2d_evaluation_audit_'+q,project,['coupled_audit.py','evaluation'],[target/'evaluation_audit.json'],
                inputs=source_inputs+artifacts(target/'dataset.npz',target/'test_table.npz')+[target/'test.json',target/'selection.json',target/'models'])
    def report_inputs(self):
        paths=[self.root/'adc_runs.json',self.adc/'checkpoints',self.adc/'data',
               self.d2d/'results/coupled_v1/independent_oracle.json']
        for run in json.loads((self.root/'adc_runs.json').read_text()):
            paths.extend(self.root/run['path']/name for name in ('trace.json','confirmation.json'))
        for q in QUALITIES:
            base=self.root/'d2d_qualities'/q/'results/coupled_v1'
            paths.extend(base/name for name in ('config.json','test.json','dataset.json','selection.json','validation.json'))
        return paths


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
        campaign.run('final_report',root,['campaign_report.py','--root',root],[root/'REPORT.md',root/'summary.json'],inputs=campaign.report_inputs())
        campaign.event('completed')
    except Exception as e:
        campaign.event('failed',record=dict(error=repr(e)))
        raise

if __name__=='__main__':main()
