
import numpy as np

from c3x.enomo.models import EnergyStorage, EnergySystem, Tariff, Demand, Generation, DemandTariff
from c3x.enomo.energy_optimiser import BTMEnergyOptimiser, OptimiserObjective
from settings import N_INTERVALS
import pytest


from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis import given, settings


@pytest.mark.parametrize(
    "minimum_demand,demand",
    [
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
        (2.0, 2.0),
    ]
)
@pytest.mark.parametrize(
    "battery_capacity",
    [1.0, 2.0, 3.0, 4.0, 5.0]
)
def test_system_precharges_for_demand_tariff(minimum_demand, demand, battery_capacity):
    """
    Test that we appropriately minimise the demand charge associated with the demand period.
    The tests use the minimal objective function that gives a well-defined result,
    which in this case is the combination of DemandCharges and ThroughputCost.
    ThroughputCost is required to prevent unnecessary battery cycling.
    """

    battery = EnergyStorage(
        max_capacity=battery_capacity,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.1,  # This must be less than the demand charge * number of periods
    )
    tariff = Tariff(
        import_tariff=dict(enumerate([1.0] * 48)),
        export_tariff=dict(enumerate([1.0] * 48))
    )
    active_periods = [0] * 24 + [1] * 12 + [0] * 12
    demand_tariff = DemandTariff(
        active_periods=dict(enumerate(active_periods)),
        cost=10.0,
        minimum_demand=minimum_demand
    )
    energy_system = EnergySystem(energy_storage=battery, export_limit=3.0, tariff=tariff,
        generation=Generation(np.array([0.0] * 48)),
        demand=Demand(np.array([demand] * 48)),
        demand_tariff=demand_tariff
    )

    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, [
            OptimiserObjective.DemandCharges,
            OptimiserObjective.ThroughputCost
        ],
    )
    df = optimiser.result_df()

    
    # Check that we reduce minimum demand appropriately during the demand period
    np.testing.assert_array_almost_equal(df.btm_net_import.values[24:36], np.ones(12) * max(demand - battery_capacity / 12, min(minimum_demand, demand)))
    np.testing.assert_array_almost_equal(df.storage_discharge_total.values[24:36],
                                         np.ones(12) * max(-battery_capacity / 12, min(minimum_demand - demand, 0.0)))




@pytest.mark.slow
@settings(deadline=3000)
@given(arrays(float, 12, elements=floats(1, 100)))
def test_demand_charge_minimised_given_random_demand_in_period(demand_period_demand):
    # TODO Given the way in which hypothesis uncovers all kinds of issues,
    # we should generate a strategy
    demand_period_demand = [1.        , 1.00163636, 1.00163636, 1.00163636, 1.00163636,
           1.00163636, 1.00163636, 1.00163636, 1.00163636, 1.00163636,
           1.00163636, 1.00163636]
    minimum_demand = 0.0
    demand = np.concatenate([np.zeros(24), demand_period_demand, np.ones(12)])
    battery_capacity = 1000.0
    battery_power = 2.0
    battery = EnergyStorage(
        max_capacity=battery_capacity,
        depth_of_discharge_limit=0,
        charging_power_limit=battery_power,
        discharging_power_limit=-battery_power,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.1,  # This must be less than the demand charge * number of periods
    )
    tariff = Tariff(
        import_tariff=dict(enumerate([1.0] * 48)),
        export_tariff=dict(enumerate([1.0] * 48))
    )
    active_periods = [0] * 24 + [1] * 12 + [0] * 12
    demand_tariff = DemandTariff(
        active_periods=dict(enumerate(active_periods)),
        cost=10.0,
        minimum_demand=minimum_demand
    )
    energy_system = EnergySystem(energy_storage=battery, export_limit=3.0, tariff=tariff,
                                 generation=Generation(np.array([0.0] * 48)),
                                 demand=Demand(demand),
                                 demand_tariff=demand_tariff
                                 )

    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, [
            OptimiserObjective.DemandCharges,
            OptimiserObjective.ThroughputCost
        ],
    )
    df = optimiser.result_df()

    # Check that we reduce minimum demand appropriately during the demand period
    max_gross_demand = max(demand_period_demand)
    max_net_demand = max(0.0, max_gross_demand - battery_power / 2.0)

    expected_import = np.maximum(np.subtract(demand_period_demand, battery_power / 2.0), max_net_demand)
    expected_discharge = expected_import - demand_period_demand
    np.testing.assert_array_almost_equal(df.btm_net_import.values[24:36],
                                         expected_import,
                                         3)
    np.testing.assert_array_almost_equal(df.storage_discharge_total.values[24:36],
                                         expected_discharge,
                                         3
                                         )
