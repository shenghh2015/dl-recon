#!/bin/bash

# This script runs a series of checks to confirm that the singularity container
# can be used on run Tensorflow on GPUs on the Radiology cluster

# Check that the /scratch directory is visible within the container
echo "Listing /scratch contents..."
echo "===================================="
ls /scratch/$USER
echo ""

# Check that the environment variables have been set correctly
echo "PATH: " $PATH
echo ""
echo "LD_LIBRARY_PATH: " $LD_LIBRARY_PATH
echo ""

# Check that Tensorflow can be imported and the GPUs load


