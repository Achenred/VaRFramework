
for seed in 0 1 2 3 4 5 6 7 8 9
do
 for env in "riverswim" "inventory" "population" "population_small"
 do
     for delta in 0.05 0.15 0.30
     do
                sbatch --time=04-01:00:00  --cpus-per-task=2 --ntasks-per-node=1 --mem-per-cpu=5000  --partition=longq  test.sh  ${env} ${seed} ${delta}
            done
        done
done