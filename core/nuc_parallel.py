import multiprocessing
import os
import subprocess


# #   Globals  # #

MAX_CORES = multiprocessing.cpu_count()


# #  Job execution  # #    
    
def run(target_func, job_data, common_args=(), common_kw={},
        num_cpu=MAX_CORES, verbose=True):
  # This does not use Pool.apply_async because of its inability to pass
  # pickled non-global functions
  
  from multiprocessing import Queue, Process
  from nuc_util import info
  
  def _wrapper(job, out_queue, target_func, data_item, args, kw):
    result = target_func(data_item, *args, **kw)
    out_queue.put((job, result))

  num_jobs = len(job_data)
  num_proc = min(num_cpu, num_jobs)
  procs = {} # Current processes
  queue = Queue() # Collect output
  results = [None] * num_jobs
  
  if verbose:
    msg = 'Running %s for %d tasks on %d cores'
    info(msg % (target_func.__name__, num_jobs, num_proc))
  
  k = 0
  for j in range(num_jobs):
    
    if len(procs) == num_proc: # Full
      i, result = queue.get()  # Async wait
      results[i] = result
      del procs[i]
      
      if verbose:
        k += 1
        msg = ' .. done %d of %d'
        info(msg % (k, num_jobs))
        
    args = (j, queue, target_func, job_data[j], common_args, common_kw)
    proc = Process(target=_wrapper, args=args)
    procs[j] = proc
    proc.start()
  
  # Last waits
  
  while procs:
    i, result = queue.get()
    results[i] = result
    del procs[i]
   
    if verbose:
      k += 1
      msg = ' .. done %d of %d'
      info(msg % (k, num_jobs))
 
  queue.close()
 
  return results
  

def parallel_split_job(target_func, split_data, common_args, num_cpu=MAX_CORES, collect_output=True):
  
  num_tasks   = len(split_data)
  num_process = min(num_cpu, num_tasks)
  processes   = []
  
  def _parallel_func_wrapper(queue, target_func, proc_data, common_args):
 
    for t, data_item in proc_data:
      result = target_func(data_item, *common_args)
 
      if queue:
        queue.put((t, result))
  
  if collect_output:
    queue = multiprocessing.Queue() # Queue will collect parallel process output
  
  else:
    queue = None
    
  for p in range(num_process):
    # Task IDs and data for each task
    # Each process can have multiple tasks if there are more tasks than processes/cpus
    proc_data = [(t, data_item) for t, data_item in enumerate(split_data) if t % num_cpu == p]
    args = (queue, target_func, proc_data, common_args)

    proc = multiprocessing.Process(target=_parallel_func_wrapper, args=args)
    processes.append(proc)
  
  for proc in processes:
    proc.start()
  
  
  if queue:
    results = [None] * num_tasks
    
    for i in range(num_tasks):
      t, result = queue.get() # Asynchronous fetch output: whichever process completes a task first
      results[t] = result
 
    queue.close()
 
    return results
  
  else:
    for proc in processes: # Asynchromous wait and no output captured
      proc.join()
    
 
