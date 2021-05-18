import os
for fac in [0.0,0.1,0.2,0.3,0.4,0.5]:
    os.system('python3 plotting-multi.py --indir ./logdir/track-transfer-cheetah-2task/'+str(fac)+'  --outdir ./plots/cheetahbul-2task/'+str(fac)+'/ --xaxis step --yaxis HalfCheetahBulletEnv-v0/test/return --bins 3e4 --xlim 0 1e6 --ylim -1000 3000 --xlabel "Environment Steps       1e6" --ylabel "Episode Return" --title '+str(fac)+' --singledir ./logdir/track-cheetah-bullet-base')
#    os.system('python3 plotting.py --indir ./logdir/track-transfer-cheetah-2task/'+str(fac)+'  --outdir ./plots/cheetahbul-2task/'+str(fac)+'/ --xaxis step --yaxis HalfCheetahBulletEnv-v0/test/return --bins 3e4 --xlim 0 1e6 --ylim -1000 3000 --xlabel "Environment Steps       1e6" --ylabel "Episode Return" --title Bullet-Cheetah-Base-'+str(fac))
    
 #   python3 plotting-multi.py --indir ./logdir/track-transfer-cheetah/0.35  --outdir ./plots/cheetahbul/0.35/ --xaxis step --yaxis HalfCheetahBulletEnv-v0/test/return --bins 3e4 --xlim 0 1e6 --ylim -1000 3000 --xlabel "Environment Steps       1e6" --ylabel "Episode Return" --title Bullet-Cheetah-Base-0.35 --singledir ./logdir/track-cheetah-bullet-base

