
import numpy as np

from c3x.enomo.models import EnergyStorage, EnergySystem, Tariff, Demand, Generation, CapacityPrices
from c3x.enomo.energy_optimiser import BTMEnergyOptimiser, OptimiserObjectiveSet, OptimiserObjective
from settings import flat_generation, flat_load, flat_tariff, N_INTERVALS, to_dict
import pytest


def test_system_charges_to_bid_into_fcas_raise():
    """
    Test that a system will charge initially in order to bid into FCAS raise services.
    The throughput cost (at twice the FCAS Raise value) ensures we don't continue to charge
    in order to bid additional into FCAS raise
    """
    battery = EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=2.0,
    )
    tariff = Tariff(
        import_tariff=dict(enumerate([1.0] * 48)),
        export_tariff=dict(enumerate([0.5] * 48))
    )
    fcas_prices = CapacityPrices(
        charge_prices=dict(enumerate([0.0] * 48)),
        discharge_prices=dict(enumerate([1.0] * 48)),
    )
    energy_system = EnergySystem(energy_storage=battery, export_limit=3.0, tariff=tariff,
        generation=Generation(np.array([0.0] * 48)),
        demand=Demand(np.array([0.0] * 48)),
        capacity_prices=fcas_prices,
    )

    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FCASOptimisation, 
    )
    df = optimiser.result_df()

    
    # Check we charge in the first interval
    np.testing.assert_array_almost_equal(df.storage_charge_total.values, np.array([1.0 / 3.0] + [0.0] * 47))

    # We can bid a little extra into the first period
    np.testing.assert_array_almost_equal(df.fcas_discharge_power.values, np.array([-2.0 - 2.0 / 3.0] + [-2.0] * 47))
    