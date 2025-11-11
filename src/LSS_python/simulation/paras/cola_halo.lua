-- cola code parameter file
nc = 1024
boxsize = 800.0

random_seed= 1111
nrealization= 1   -- multiple realisations for random_seed, random_seed+1, ...

ntimestep= 30
a_final= 1.0
output_redshifts= {1.0, 0.6069, 0.0}  -- redshifts of output

omega_m = 0.3071
h       = 0.6777
sigma8  = 0.8288
de_w = -1.0

pm_nc_factor= 3            -- Particle Mesh grid pm_nc_factor*nc per dimension
np_alloc_factor= 1.2      -- Amount of memory allocated for particle
loglevel=1                 -- 0=debug, 1=verbose, 2=normal, ...
                           -- increase the value to reduce output msgs

powerspectrum= "your/linear/power/spectrum/file" -- Initial power spectrum: k P(k)

-- Options
--   Following outputs can be turned off by commenting out

-- use_solve_growth = true -- use a new growth solution from xiaodong

--   fof, snapshot, subsample, coarse_grid

-- FoF halo catalogue
-- fof= "fof"                 -- base filename
-- linking_factor= 0.2        -- FoF linking length= linking_factor*mean_separation

-- Dark matter particle outputs (all particles)
snapshot= "your/snap/dir/to/store"       -- comment out to suppress snapshot output

-- Dark matter particle subsample
-- subsample= "sub"        -- base filename
-- subsample_factor= 0.01     -- fraction of particles to output

-- Dark matter density grid
-- coarse_grid= "grid"     -- base filename
-- coarse_grid_nc= 16         -- number of grid per dimension

-- Output 2LPT initial condition at a_final/ntimestep
-- ntimestep=1 will only generate 2LPT field with no COLA steps.
-- initial= "init"


-- Use 8-byte long id for GADGET snapshot
write_longid= true -- true or false