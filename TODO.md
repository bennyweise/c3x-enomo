# Fixes

## Export Limits
- enable export limits on local energy systems (this must be defined - is it on the storage system,
or on the aggregated system)
- additional testing - particularly that arbitrage works appropriately, and negative export prices cause curtailment


# Features to Add


## FCAS Optimisation
Initial implementation of FCAS optimisation, with additional features

## Battery / Solar size Optimisation
This will be quite a large piece of development. Given normalised solar curves, it may be possible
to optimise the price /performance of a system to find the optimal combination of solar and battery 
for a given price constraint.
It's more likely that this will work 




# Architectural Updates

## More input validation

## Easier access to results
(and integration with plotting libraries?)

## Real-time commands 
Generated based on perturbation analysis or similar


## Variable period-length optimisation


## Enforcement of Units
Or at least better documentation around the use of units, particularl power/energy conversions.
