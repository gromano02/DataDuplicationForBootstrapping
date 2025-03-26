#!/bin/bash

# Find and delete all files with "slurm" in their name
find . -type f -name "*slurm*" -exec rm -f {} +

echo "Deleted all files containing 'slurm' in the name."
