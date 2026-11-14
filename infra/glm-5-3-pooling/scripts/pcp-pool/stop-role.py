import sys,subprocess,pathlib,os,signal,time,urllib.request,re
role=sys.argv[1]
assert role in ('p','d','proxy','meta')
base='/home/d00937186/glm53-pooling-lab'
patterns={'p':['vllm serve /home/weights/GLM-5.3-w4a4c8-mxfp4','--port 8100',base+'/profiling/p','MultiConnector'],'d':['vllm serve /home/weights/GLM-5.3-w4a4c8-mxfp4','--port 8200',base+'/profiling/d','MultiConnector'],'proxy':[base+'/scripts/pcp-pool/load_balance_proxy_server_example.py','--port 9000'],'meta':['python3 -c from memcache_hybrid import MetaService; MetaService.main()']}
if role in ('p','d'):
 port=8100 if role=='p' else 8200
 with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/metrics',timeout=10) as r: metrics=r.read().decode()
 rows=[x for x in metrics.splitlines() if x.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
 assert rows and all(float(x.rsplit(' ',1)[1])==0 for x in rows),'in-flight requests or missing metrics'
rows=subprocess.check_output(['docker','top','vllm-pool','-eo','pid,args'],text=True).splitlines()[1:]
matches=[int(x.split()[0]) for x in rows if all(t in x for t in patterns[role])]
assert len(matches)==1,('expected one owned process',matches)
pid=matches[0]
if role=='meta':
 env=pathlib.Path('/proc/'+str(pid)+'/environ').read_bytes()
 assert b'MMC_META_CONFIG_PATH=/home/d00937186/glm53-pooling-lab/config/l3/mmc-meta.conf' in env
print('Stopping',role,pid,flush=True)
os.kill(pid,signal.SIGTERM)
for _ in range(60):
 if not pathlib.Path('/proc/'+str(pid)).exists(): break
 time.sleep(1)
else: raise RuntimeError('Process did not exit; inspect manually, no SIGKILL applied')
print('Stopped',role,pid)
