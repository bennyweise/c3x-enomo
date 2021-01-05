# Fixes

## Export Limits
- enable export limits on local energy systems (this must be defined - is it on the storage system,
or on the aggregated system)
- additional testing - particularly that arbitrage works appropriately, and negative export prices cause curtailment

## FCAS Optimisation
- Currently bids are based on the amount of energy available at the end of a period, where technically
we should probably treat the constraint as the minimum energy available during the period to ensure
the availability can be met throughout the period.
- Test that FCAS bids don't exceed site export limits
- The capacity availability constraints, if included by default, cause many BTM optimisation tests to fail.
This hasn't been investigated, so these constraints are only included when the objective includes 
`CapacityAvailability`. This needs to be investigated and fixed.

# Features to Add




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

## Demand / Generation units
Allow for demand and generation to actually reflect real-world load and solar, 
rather than having the requirement that only one may be non-zero at every timestep
