 #!/bin/bash

 echo "/mnt/nfs/home/elitalobo/aggrxp/datasets/${1}/config.yml"

 python /mnt/nfs/home/elitalobo/rl/robustrl/src/main_swarm.py --env ${1} --seed ${2} --delta ${3}
