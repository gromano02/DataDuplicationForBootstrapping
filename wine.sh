#!/bin/bash
# Define methods and factors
methods=('undersampling' 'class_weighted' 'standard')
factors=(1 2 3 4 5 10 15 20)

# Number of separate jobs per (method, factor) combination
num_repeats=20

# Loop over methods and factors
for method in "${methods[@]}"; do
    for factor in "${factors[@]}"; do
        for repeat in $(seq 1 $num_repeats); do
            # Create a unique sbatch script for each method-factor-repeat combination
            sbatch_script="wine_${method}_${factor}_rep${repeat}.sbatch"

            echo "#!/bin/bash
#SBATCH --partition=priority-gpu
#SBATCH --qos=qiy18011a100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=5G

module purge
module load python

echo \"Running: method=$method, factor=$factor, iteration=$repeat\"
python3 wine.py --method $method --factor $factor
" > "$sbatch_script"

            # Submit the sbatch script
            sbatch "$sbatch_script" && rm "$sbatch_script"
        done
    done
done

