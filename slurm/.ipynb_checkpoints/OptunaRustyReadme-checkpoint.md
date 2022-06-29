# Optuna on rusty with postgres

These two files demonstrate how to use postgres as a database backend on rusty to coordinate optuna jobs.
This is significantly more reliable than using sqlite with multiple writers on a shared filesystem.

We first start the database script with:
```
sbatch run_db.sh
```
This will setup a database job, and create the default database stuff.

Once the job is started, we can ask slurm to check where it is running:
```
> squeue -u $USER

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1436762      genx run_db.s    wzhou  R       0:04      1 worker1172
```
We know that we should connect on worker1172.
We can now run our python script from anywhere (e.g. desktop, or another slurm job).
```
python script.py --host worker1172 --port 35719
```
The current script will run 20 trials, coordinating with all other past / concurrent runs.

Once we are done, we may simply cancel the database job.
Before that, it might be helpful to connect through optuna and print out the current study info (e.g. by using the `trials_dataframe` framework.
However, in case the database job was interrupted for some other reason, or we forgot to save the study info, all the database information is also saved in `backup.gz`.
This backup can be used to restore the entire database info if necessary.
It is somewhat more cumbersome to work with than the simply dataframe obtained from the study object in python.

