
import numpy as np

from c3x.enomo.models import EnergyStorage, EnergySystem, Tariff, Demand, Generation
from c3x.enomo.energy_optimiser import BTMEnergyOptimiser, OptimiserObjectiveSet, OptimiserObjective
from settings import flat_generation, flat_load, flat_tariff, N_INTERVALS, to_dict
import pytest

@pytest.mark.parametrize(
    "export_limit,generation",
    [(1.0, 10.0), (5.0, 10.0), (10.0, 10.0), (15.0, 10.0)],
)
def test_curtailment_to_export_limit(export_limit: float, generation: float):
    """
    Test that we limit generation based on the `export_limit`
    """
    battery = EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=0.9,
        discharging_efficiency=0.9,
        throughput_cost=0.0,
    )
    energy_system = EnergySystem(
        energy_storage=battery,
        demand=Demand(flat_load * 0.0),
        generation=Generation(flat_generation * generation),
        tariff=Tariff(
            import_tariff=dict(enumerate(flat_tariff)),
            export_tariff=dict(enumerate(flat_tariff))
        ),
        export_limit=export_limit
    )
    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FinancialOptimisation, 
    )

    np.testing.assert_array_equal(
        optimiser.values("system_generation"), np.ones(N_INTERVALS) * -min(export_limit, generation)
    )


def test_battery_charges_preferentially_from_curtailed_solar():
    """
    Test that, in a simplified system, the battery will preferentially charge from
    generation that would otherwise need to be curtailed.
    """
    battery = EnergyStorage(
        max_capacity=4.0,
        depth_of_discharge_limit=0,
        charging_power_limit=2.0,
        discharging_power_limit=-2.0,
        charging_efficiency=1.0,
        discharging_efficiency=1.0,
        throughput_cost=0.0,
    )
    tariff = Tariff(
        import_tariff=dict(enumerate([1.0] * 48)),
        export_tariff=dict(enumerate([0.5] * 48))
    )
    energy_system = EnergySystem(energy_storage=battery, export_limit=3.0, tariff=tariff,
        generation=Generation(np.array([-5.0] * 12 + [-3.0] * 12 + [0.0] * 24)),
        demand=Demand(np.array([0.0] * 24 + [5.0] * 24)),
    )

    optimiser = BTMEnergyOptimiser(
        30, N_INTERVALS, energy_system, OptimiserObjectiveSet.FinancialOptimisation, 
    )
    df = optimiser.result_df()

    np.testing.assert_array_almost_equal(df.system_generation.values, np.array([-3.0 - 4.0 / 12] * 12 + [-3.0] * 12 + [0.0] * 24))
    np.testing.assert_array_almost_equal(df.storage_charge_total.values, np.array([4.0 / 12] * 12 + [0.0] * 36))
    np.testing.assert_array_almost_equal(df.storage_discharge_total.values, np.array([0.0] * 24 + [-4.0 / 24] * 24))
    