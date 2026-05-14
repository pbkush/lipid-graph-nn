#!/bin/bash
# Wrapper so the martini pipeline can invoke `gmx_mpi` as if it were `gmx`.
#
# pipeline.run() uses a single `gmx_executable` for both grompp and mdrun, but
# on Goethe-HLR's general1 partition only `gmx_mpi` is installed (spack
# gromacs/2022.4-gcc-11.3.1-zx2wwcx).  gmx_mpi requires `mpirun` even for
# single-rank invocations under openmpi 5.x.
#
# This wrapper makes every pipeline gmx call run as `mpirun -np 1 gmx_mpi ...`.
# grompp gets a small MPI-init overhead (single rank, no comm cost); mdrun
# runs single-rank and parallelises via OpenMP threads (-ntomp via --mdrun-args).
#
# Requires the calling environment to have loaded:
#   module load mpi/openmpi/5.0.0
#   module load gromacs/2022.4-gcc-11.3.1-zx2wwcx
#
# Invoked by sbatch_setup_general1.sh as the pipeline's --gmx argument.
exec mpirun -np 1 gmx_mpi "$@"
