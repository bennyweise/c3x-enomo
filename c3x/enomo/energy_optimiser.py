from pyomo.opt import SolverFactory
import pyomo.environ as en
import os
import numpy as np

####################################################################

# Define some useful constants
minutes_per_hour = 60.0
fcas_6s_duration = 1.0
fcas_60s_duration = 4.0
fcas_5m_duration = 5.0
fcas_total_duration = fcas_6s_duration + fcas_60s_duration + fcas_5m_duration


# Define some useful container objects to define the optimisation objectives

class OptimiserObjective(object):
    ConnectionPointCost = 1
    ConnectionPointEnergy = 2
    ThroughputCost = 3
    Throughput = 4
    GreedyGenerationCharging = 5
    GreedyDemandDischarging = 6
    EqualStorageActions = 7
    ConnectionPointPeakPower = 8
    ConnectionPointQuantisedPeak = 9
    PiecewiseLinear = 10
    LocalModelsCost = 11
    LocalGridMinimiser = 12
    LocalThirdParty = 13
    LocalGridPeakPower = 14

    CapacityAvailability = 15
    
class OptimiserObjectiveSet(object):
    FinancialOptimisation = [OptimiserObjective.ConnectionPointCost,
                             #OptimiserObjective.GreedyGenerationCharging,
                             OptimiserObjective.ThroughputCost,
                             OptimiserObjective.EqualStorageActions
                             ]

    FCASOptimisation = FinancialOptimisation + [OptimiserObjective.CapacityAvailability]                             

    EnergyOptimisation = [OptimiserObjective.ConnectionPointEnergy,
                          OptimiserObjective.GreedyGenerationCharging,
                          OptimiserObjective.GreedyDemandDischarging,
                          OptimiserObjective.Throughput,
                          OptimiserObjective.EqualStorageActions]

    PeakOptimisation = [OptimiserObjective.ConnectionPointPeakPower]

    QuantisedPeakOptimisation = [OptimiserObjective.ConnectionPointQuantisedPeak]

    DispatchOptimisation = [OptimiserObjective.PiecewiseLinear] + FinancialOptimisation

    LocalModels = [OptimiserObjective.LocalModelsCost, 
                   OptimiserObjective.ThroughputCost,
                   OptimiserObjective.EqualStorageActions]

    LocalModelsThirdParty = [OptimiserObjective.LocalThirdParty,
                             OptimiserObjective.ThroughputCost,
                             OptimiserObjective.EqualStorageActions]

    LocalPeakOptimisation = [OptimiserObjective.LocalGridPeakPower]


####################################################################

class EnergyOptimiser(object):
    
    def __init__(self, interval_duration, number_of_intervals, energy_system, objective):
        self.interval_duration = interval_duration  # The duration (in minutes) of each of the intervals being optimised over
        self.number_of_intervals = number_of_intervals
        self.energy_system = energy_system

        # Configure the optimiser through setting appropriate environmental variables.
        self.optimiser_engine = os.environ.get('OPTIMISER_ENGINE', '_cplex_shell')  # ipopt doesn't work with int/bool variables
        self.optimiser_engine_executable = os.environ.get('OPTIMISER_ENGINE_EXECUTABLE')

        self.use_bool_vars = True

        # These values have been arbitrarily chosen
        # A better understanding of the sensitivity of these values may be advantageous
        self.bigM = 5000000
        self.smallM = 0.0001

        self.objectives = objective
        self.build_model()
        self.apply_constraints()
        self.build_objective()

        
    def build_model(self):
        # Set up the Pyomo model
        self.model = en.ConcreteModel()

        # We use RangeSet to create a index for each of the time
        # periods that we will optimise within.
        self.model.Time = en.RangeSet(0, self.number_of_intervals - 1)

        # Configure the initial demand and generation
        system_demand = self.energy_system.demand.demand
        system_generation = self.energy_system.generation.generation
        # and convert the data into the right format for the optimiser objects
        self.system_demand_dct = dict(enumerate(system_demand))
        self.system_generation_dct = dict(enumerate(system_generation))

        #### Initialise the optimisation variables (all indexed by self.model.Time) ####

        # The state of charge of the battery
        self.model.storage_state_of_charge = en.Var(self.model.Time,
                                                    bounds=(0, self.energy_system.energy_storage.capacity),
                                                    initialize=0)

        # The increase in energy storage state of charge at each time step
        self.model.storage_charge_total = en.Var(self.model.Time, initialize=0)

        # The decrease in energy storage state of charge at each time step
        self.model.storage_discharge_total = en.Var(self.model.Time, initialize=0)

        # Increase in battery SoC from the Grid
        self.model.storage_charge_grid = en.Var(self.model.Time,
                                                bounds=(0, self.energy_system.energy_storage.charging_power_limit *
                                                        (self.interval_duration / minutes_per_hour)),
                                                initialize=0)

        # Increase in battery SoC from PV Generation
        self.model.storage_charge_generation = en.Var(self.model.Time,
                                                      bounds=(0, self.energy_system.energy_storage.charging_power_limit *
                                                                 (self.interval_duration / minutes_per_hour)),
                                                      initialize=0)

        # Satisfying local demand from the battery
        self.model.storage_discharge_demand = en.Var(self.model.Time,
                                                     bounds=(self.energy_system.energy_storage.discharging_power_limit *
                                                           (self.interval_duration / minutes_per_hour), 0),
                                                     initialize=0)

        # Exporting to the grid from the battery
        self.model.storage_discharge_grid = en.Var(self.model.Time,
                                                   bounds=(self.energy_system.energy_storage.discharging_power_limit *
                                                           (self.interval_duration / minutes_per_hour), 0),
                                                   initialize=0)

        #### Boolean variables (again indexed by Time) ####

        # These may not be necessary so provide a binary flag for turning them off

        if self.use_bool_vars:
            # Is the battery charging in a given time interval
            self.model.is_charging = en.Var(self.model.Time, within=en.Boolean)
            # Is the battery discharging in a given time interval
            self.model.is_discharging = en.Var(self.model.Time, within=en.Boolean, initialize=0)

            self.model.local_demand_satisfied = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            self.model.local_generation_satisfied = en.Var(self.model.Time, within=en.Boolean, initialize=0)

            self.model.is_importing = en.Var(self.model.Time, within=en.Boolean)
            # Is the battery discharging in a given time interval
            self.model.is_local_exporting = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            

        #### Battery Parameters ####

        # The battery charging efficiency
        self.model.eta_chg = en.Param(initialize=self.energy_system.energy_storage.charging_efficiency)
        # The battery discharging efficiency
        self.model.eta_dischg = en.Param(initialize=self.energy_system.energy_storage.discharging_efficiency)
        # The battery charge power limit
        self.model.charging_limit = en.Param(
            initialize=self.energy_system.energy_storage.charging_power_limit * (self.interval_duration / minutes_per_hour))
        # The battery discharge power limit
        self.model.discharging_limit = en.Param(
            initialize=self.energy_system.energy_storage.discharging_power_limit * (self.interval_duration / minutes_per_hour))
        # The throughput cost for the energy storage
        self.model.throughput_cost = en.Param(initialize=self.energy_system.energy_storage.throughput_cost)

        #### Bias Values ####
        
        # A small fudge factor for reducing the size of the solution set and
        # achieving a unique optimisation solution
        self.model.scale_func = en.Param(initialize=self.smallM)
        # A bigM value for integer optimisation
        self.model.bigM = en.Param(initialize=self.bigM)

        #### Initial Demand / Generation Profile Parameters ####
        
        # The local energy consumption
        self.model.system_demand = en.Param(self.model.Time, initialize=self.system_demand_dct)
        # The local energy generation
        self.model.system_generation_max = en.Param(self.model.Time, initialize=self.system_generation_dct)
        self.model.system_generation = en.Var(self.model.Time, initialize=self.system_generation_dct)
        
        self.model.export_limit = en.Param(initialize=self.energy_system.export_limit)


    def apply_constraints(self):

        # Calculate the increased state of charge of the energy storage from the
        # imported energy and locally generated energy. We ensure that the
        # storage charging efficiency is taken into account.
        def storage_charge_behaviour(model, time_interval):
            return model.storage_charge_grid[time_interval] + model.storage_charge_generation[time_interval] \
                   == model.storage_charge_total[time_interval] / model.eta_chg

        # Calculate the decreased state of charge of the energy storage from the
        # exported energy and locally consumed energy. We ensure that the
        # storage discharging efficiency is taken into account.
        def storage_discharge_behaviour(model, time_interval):
            return model.storage_discharge_demand[time_interval] + model.storage_discharge_grid[time_interval] \
                   == model.storage_discharge_total[time_interval] * model.eta_dischg

        # Enforce the charging rate limit
        def storage_charge_rate_limit(model, time_interval):
            return (model.storage_charge_grid[time_interval] + model.storage_charge_generation[
                time_interval]) <= model.charging_limit

        # Enforce the discharge rate limit
        def storage_discharge_rate_limit(model, time_interval):
            return (model.storage_discharge_demand[time_interval] + model.storage_discharge_grid[
                time_interval]) >= model.discharging_limit

        # Add the constraints to the optimisation model
        self.model.storage_charge_behaviour_constraint = en.Constraint(self.model.Time, rule=storage_charge_behaviour)
        self.model.storage_discharge_behaviour_constraint = en.Constraint(self.model.Time, rule=storage_discharge_behaviour)
        self.model.storage_charge_rate_limit_constraint = en.Constraint(self.model.Time, rule=storage_charge_rate_limit)
        self.model.storage_discharge_rate_limit_constraint = en.Constraint(self.model.Time, rule=storage_discharge_rate_limit)

        # Calculate the state of charge of the battery in each time interval
        initial_state_of_charge = self.energy_system.energy_storage.initial_state_of_charge

        def SOC_rule(model, time_interval):
            if time_interval == 0:
                return model.storage_state_of_charge[time_interval] \
                       == initial_state_of_charge + model.storage_charge_total[time_interval] + \
                       model.storage_discharge_total[
                           time_interval]
            else:
                return model.storage_state_of_charge[time_interval] \
                       == model.storage_state_of_charge[time_interval - 1] + model.storage_charge_total[time_interval] + \
                       model.storage_discharge_total[time_interval]

        self.model.Batt_SOC = en.Constraint(self.model.Time, rule=SOC_rule)

        # Use bigM formulation to ensure that the battery is only charging or discharging in each time interval
        if self.use_bool_vars:

            # If the battery is charging then the charge energy is bounded from below by -bigM
            # If the battery is discharging the charge energy is bounded from below by zero
            def bool_cd_rule_one(model, time_interval):
                return model.storage_charge_total[time_interval] >= -self.model.bigM * model.is_charging[time_interval]

            # If the battery is charging then the charge energy is bounded from above by bigM
            # If the battery is discharging the charge energy is bounded from above by zero
            def bool_cd_rule_two(model, time_interval):
                return model.storage_charge_total[time_interval] <= self.model.bigM * (1 - model.is_discharging[time_interval])

            # If the battery is charging then the discharge energy is bounded from above by zero
            # If the battery is discharging the discharge energy is bounded from above by bigM
            def bool_cd_rule_three(model, time_interval):
                return model.storage_discharge_total[time_interval] <= self.model.bigM * model.is_discharging[time_interval]

            # If the battery is charging then the discharge energy is bounded from below by zero
            # If the battery is discharging the discharge energy is bounded from below by -bigM
            def bool_cd_rule_four(model, time_interval):
                return model.storage_discharge_total[time_interval] >= -self.model.bigM * (1 - model.is_charging[time_interval])

            # The battery can only be charging or discharging
            def bool_cd_rule_five(model, time_interval):
                return model.is_charging[time_interval] + model.is_discharging[time_interval] == 1

            # Add the constraints to the optimisation model
            self.model.bcdr_one = en.Constraint(self.model.Time, rule=bool_cd_rule_one)
            self.model.bcdr_two = en.Constraint(self.model.Time, rule=bool_cd_rule_two)
            self.model.bcdr_three = en.Constraint(self.model.Time, rule=bool_cd_rule_three)
            self.model.bcdr_four = en.Constraint(self.model.Time, rule=bool_cd_rule_four)
            self.model.bcdr_five = en.Constraint(self.model.Time, rule=bool_cd_rule_five)

            

    def build_objective(self):
        # Build the objective function ready for optimisation
        self.objective = 0

        if OptimiserObjective.ThroughputCost in self.objectives:
            # Throughput cost of using energy storage - we attribute half the cost to charging and half to discharging
            self.objective += sum(self.model.storage_charge_total[i] - self.model.storage_discharge_total[i]
                             for i in self.model.Time) * self.model.throughput_cost / 2.0

        if OptimiserObjective.Throughput in self.objectives:
            # Throughput of using energy storage - it mirrors the throughput cost above
            self.objective += sum(self.model.storage_charge_total[i] - self.model.storage_discharge_total[i]
                             for i in self.model.Time) * self.model.scale_func

        if OptimiserObjective.EqualStorageActions in self.objectives:
            # ToDo - Which is the better implementation?
            self.objective += sum((self.model.storage_charge_grid[i] * self.model.storage_charge_grid[i]) +
                             (self.model.storage_charge_generation[i] * self.model.storage_charge_generation[i]) +
                             (self.model.storage_discharge_grid[i] * self.model.storage_discharge_grid[i]) +
                             (self.model.storage_discharge_demand[i] * self.model.storage_discharge_demand[i])
                             for i in self.model.Time) * self.model.scale_func

            '''objective += sum(self.model.storage_charge_total[i] * self.model.storage_charge_total[i] +
                             self.model.storage_discharge_total[i] * self.model.storage_discharge_total[i]
                             for i in self.model.Time) * self.model.scale_func'''

        '''if OptimiserObjective.PiecewiseLinear in self.objectives: # ToDo - Fix this implementation to make it complete
            for i in self.energy_system.dispatch.linear_ramp[0]:
                objective += -1 * (self.model.storage_charge_total[i] + self.model.storage_discharge_total[i]) * (
                        1 - self.model.turning_point_two_ramp[i])'''

    def optimise(self):
        def objective_function(model):
            return self.objective

        self.model.total_cost = en.Objective(rule=objective_function, sense=en.minimize)

        # set the path to the solver
        if self.optimiser_engine == 'cplex':
            opt = SolverFactory(self.optimiser_engine, executable=self.optimiser_engine_executable)
        else:
            opt = SolverFactory(self.optimiser_engine)

        # Solve the optimisation
        opt.solve(self.model)

    def values(self, variable_name, decimal_places=3):
        output = np.zeros(self.number_of_intervals)
        var_obj = getattr(self.model, variable_name)
        for index in var_obj:
            try:
                output[index] = round(var_obj[index].value, decimal_places)
            except AttributeError:
                output[index] = round(var_obj[index], decimal_places)
        return output

    def result_dct(self, include_indexed_params=True):
        """Extract the resulting `Var`s (and input `Param`s) as a dictionary

        Args:
            include_indexed_params (bool, optional): Whether to include indexed `Param`s in output. Defaults to True.

        Returns:
            dict: Results dict
        """
        if include_indexed_params:
            component_objects = (en.Var, en.Param)
        else:
            component_objects = en.Var
        dct = {}
        
        for var_obj in self.model.component_objects(component_objects):
            if var_obj.is_indexed():
                dct[var_obj.name] = var_obj.extract_values()

        return dct

    def result_df(self, include_indexed_params=True):
        """Return result (and optionally indexed `Param`s) as a dataframe

        Args:
            include_indexed_params (bool, optional): Whether to include indexed `Param`s in output. Defaults to True.

        Returns:
            pd.DataFrame: Results dataframe
        """
        import pandas as pd  # TODO Check if pandas is otherwise required and import at head of file
        return pd.DataFrame(self.result_dct(include_indexed_params))

class BTMEnergyOptimiser(EnergyOptimiser):

    def __init__(self, interval_duration, number_of_intervals, energy_system, objective):
        super().__init__(interval_duration, number_of_intervals, energy_system, objective)
        
        self.use_piecewise_segments = True  # Defined for a future implementation of linear piecewise segments

        self.update_build_model()
        self.update_apply_constraints()
        self.update_build_objective()
        super().optimise()

    def update_build_model(self):
        #### Behind - the - Meter (BTM) Models ####

        # Net import from the grid
        self.model.btm_net_import = en.Var(self.model.Time, initialize=self.system_demand_dct)

        # Net export to the grid
        self.model.btm_net_export = en.Var(self.model.Time, initialize=self.system_generation_dct)
        
        # The import tariff per kWh
        self.model.btm_import_tariff = en.Param(self.model.Time, initialize=self.energy_system.tariff.import_tariff)
        # The export tariff per kWh
        self.model.btm_export_tariff = en.Param(self.model.Time, initialize=self.energy_system.tariff.export_tariff)

        #### BTM Connection Point Peak Power ####

        self.model.peak_connection_point_import_power = en.Var(within=en.NonNegativeReals)
        self.model.peak_connection_point_export_power = en.Var(within=en.NonNegativeReals, bounds=(None, self.energy_system.export_limit))

        def peak_connection_point_import(model, interval):
            return model.peak_connection_point_import_power >= model.btm_net_import[interval]

        def peak_connection_point_export(model, interval):
            return model.peak_connection_point_export_power >= -model.btm_net_export[interval]

        

        self.model.peak_connection_point_import_constraint = en.Constraint(self.model.Time,
                                                                           rule=peak_connection_point_import)
        self.model.peak_connection_point_export_constraint = en.Constraint(self.model.Time,
                                                                           rule=peak_connection_point_export)

        def connection_point_export_curtailment(model, interval):
            """
            Allows generally for generation to be curtailed in order to maximise the objective.
            It is assumed that the supplied system_generation_max
            """
            return model.system_generation[interval] >= model.system_generation_max[interval]
        
        self.model.system_generation_curtailment_constraint = en.Constraint(self.model.Time,
            rule=connection_point_export_curtailment
        )

        #### Piecewise Linear Segments (To be fully implemented later) ####
        '''if self.use_piecewise_segments:
            # The turning points for the piecewise linear segments
            self.model.turning_point_one_ramp = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            self.model.turning_point_two_ramp = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            lims_one = [None] * (len(net) - 1)  # ToDo - Fix this indexing
            lims_two = [None] * (len(net) - 1)  # ToDo - Fix this indexing

            ind = self.energy_system.dispatch.linear_ramp[0]
            lim = self.energy_system.dispatch.linear_ramp[1]
            for i, l in zip(ind, lim):
                lims_one[i] = l[0]
                lims_two[i] = l[1]

            lim_dct_one = dict(enumerate(lims_one))
            self.model.limits_one = en.Param(self.model.Time, initialize=lim_dct_one)

            lim_dct_two = dict(enumerate(lims_two))
            self.model.limits_two = en.Param(self.model.Time, initialize=lim_dct_two)

            self.model.my_set = en.Set(initialize=ind)
            def B1(m, s):
                return m.limits_one[s] <= m.storage_charge_total[s] + m.storage_discharge_total[s] + self.bigM * (1 - m.turning_point_one_ramp[s])

            def B2(m, s):
                return m.limits_one[s] >= m.storage_charge_total[s] + m.storage_discharge_total[s] - self.bigM * m.turning_point_one_ramp[s]

            self.model.B1 = en.Constraint(self.model.my_set, rule=B1)
            self.model.B2 = en.Constraint(self.model.my_set, rule=B2)

            def B3(m, s):
                return m.limits_two[s] <= m.storage_charge_total[s] + m.storage_discharge_total[s] + self.bigM * (1 - m.turning_point_two_ramp[s])

            def B4(m, s):
                return m.limits_two[s] >= m.storage_charge_total[s] + m.storage_discharge_total[s] - self.bigM * m.turning_point_two_ramp[s]

            self.model.B3 = en.Constraint(self.model.my_set, rule=B3)
            self.model.B4 = en.Constraint(self.model.my_set, rule=B4)'''

    def update_apply_constraints(self):

        # Enforce the limits of charging the energy storage from locally generated energy
        def storage_generation_charging_behaviour(model, time_interval):
            return model.storage_charge_generation[time_interval] <= -model.system_generation[time_interval]

        # Enforce the limits of discharging the energy storage to satisfy local demand
        def storage_demand_discharging_behaviour(model, time_interval):
            return model.storage_discharge_demand[time_interval] >= -model.system_demand[time_interval]

        # Add the constraints to the optimisation model
        self.model.generation_charging_behaviour_constraint = en.Constraint(self.model.Time,
                                                                            rule=storage_generation_charging_behaviour)
        self.model.local_discharge_behaviour_constraint = en.Constraint(self.model.Time,
                                                                        rule=storage_demand_discharging_behaviour)

        # Calculate the net energy import
        def btm_net_connection_point_import(model, time_interval):
            return model.btm_net_import[time_interval] == model.system_demand[time_interval] + \
                   model.storage_charge_grid[time_interval] + model.storage_discharge_demand[time_interval]

        # calculate the net energy export
        def btm_net_connection_point_export(model, time_interval):
            return model.btm_net_export[time_interval] == model.system_generation[time_interval] + \
                   model.storage_charge_generation[time_interval] + model.storage_discharge_grid[time_interval]

        # Add the constraints to the optimisation model
        self.model.btm_net_import_constraint = en.Constraint(self.model.Time, rule=btm_net_connection_point_import)
        self.model.btm_net_export_constraint = en.Constraint(self.model.Time, rule=btm_net_connection_point_export)

        self._apply_fcas_constraints()

    def _apply_fcas_constraints(self):
        #### Capacity Availability ####
        
        if OptimiserObjective.CapacityAvailability in self.objectives:  # We can always incorporate these but not optimiser for them.
            # Need to find the max of battery capacity / max power versus set point
            self.model.fcas_storage_discharge_power_is_max = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            self.model.fcas_storage_charge_power_is_max = en.Var(self.model.Time, within=en.Boolean, initialize=0)
            self.model.fcas_discharge_power = en.Var(self.model.Time, initialize=0, bounds=(None, 0))#, domain=en.Integers)
            self.model.fcas_charge_power = en.Var(self.model.Time, initialize=0)#, domain=en.Integers)

            def fcas_max_power_raise_rule_one(model, time_interval):
                # Keep the fcas power below the maximum available power
                return model.fcas_discharge_power[time_interval] >= (
                    self.energy_system.energy_storage.discharging_power_limit -
                        (
                            model.storage_charge_total[time_interval] / model.eta_chg 
                            + model.storage_discharge_total[time_interval] * model.eta_dischg
                        ) * minutes_per_hour / self.interval_duration
                    )

            def fcas_max_power_raise_rule_two(model, time_interval):
                # Keep the fcas power below that which can be supplied by the available energy storage
                # This needs to account for the ability of the response to stop charging (which requires no energy)
                allocated_energy = (
                    model.storage_charge_total[time_interval] / model.eta_chg 
                    + model.storage_discharge_total[time_interval] * model.eta_dischg
                ) * minutes_per_hour / self.interval_duration
                fcas_energy_required_per_bid_unit = (fcas_total_duration / minutes_per_hour) / self.model.eta_dischg
                return model.fcas_discharge_power[time_interval] >= (
                        -self.model.storage_state_of_charge[time_interval] / (fcas_energy_required_per_bid_unit)
                        -allocated_energy
                        # / (fcas_total_duration / minutes_per_hour) - allocated_energy
                    )

            def fcas_max_power_raise_rule_three(model, time_interval):
                return model.fcas_discharge_power[time_interval] <= self.energy_system.energy_storage.discharging_power_limit - (model.storage_charge_total[time_interval] + model.storage_discharge_total[time_interval]) + self.bigM * (1 - self.model.fcas_storage_discharge_power_is_max[time_interval])

            def fcas_max_power_raise_rule_four(model, time_interval):
                return model.fcas_discharge_power[time_interval] <= -self.model.storage_state_of_charge[time_interval] * self.model.eta_dischg / (fcas_total_duration / minutes_per_hour) + self.bigM * self.model.fcas_storage_discharge_power_is_max[time_interval]

            def fcas_max_power_raise_export_limit(model, time_interval):
                # Ensure that the capacity allocated to FCAS won't exceed the site export limit
                power_conversion = minutes_per_hour / self.interval_duration

                excess_charge_power = (self.model.system_generation[time_interval] + model.storage_charge_total[time_interval] + model.storage_discharge_total[time_interval]) * power_conversion
                return -model.fcas_discharge_power[time_interval] - excess_charge_power <= self.model.export_limit
                

            self.model.fcas_max_power_raise_rule_one_constraint = en.Constraint(self.model.Time,
                                                                                rule=fcas_max_power_raise_rule_one)
            self.model.fcas_max_power_raise_rule_two_constraint = en.Constraint(self.model.Time,
                                                                                rule=fcas_max_power_raise_rule_two)
            # self.model.fcas_max_power_raise_rule_three_constraint = en.Constraint(self.model.Time,
            #                                                                     rule=fcas_max_power_raise_rule_three)
            # self.model.fcas_max_power_raise_rule_four_constraint = en.Constraint(self.model.Time,
            #                                                                     rule=fcas_max_power_raise_rule_four)
            
            
            # TODO Probably a better way of doing this
            if self.energy_system.export_limit is not None:
                self.model.fcas_max_power_raise_export_limit_rule = en.Constraint(
                    self.model.Time,
                    rule=fcas_max_power_raise_export_limit
                )

            def fcas_max_power_lower_rule_one(model, time_interval):
                # Keep the fcas power below the maximum available power
                
                return model.fcas_charge_power[time_interval] <= (
                    self.energy_system.energy_storage.charging_power_limit - 
                        (
                            model.storage_charge_total[time_interval] / model.eta_chg 
                            + model.storage_discharge_total[time_interval] * model.eta_dischg
                        ) * minutes_per_hour / self.interval_duration
                    )

            def fcas_max_power_lower_rule_two(model, time_interval):
                # Keep the fcas power below that which can be supplied by the available energy storage
                allocated_energy = (
                    model.storage_charge_total[time_interval] / model.eta_chg 
                    + model.storage_discharge_total[time_interval] * model.eta_dischg
                ) * minutes_per_hour / self.interval_duration
                return model.fcas_charge_power[time_interval] <= (
                        self.energy_system.energy_storage.max_capacity - self.model.storage_state_of_charge[time_interval]
                    ) / (self.model.eta_chg * fcas_total_duration / minutes_per_hour) + allocated_energy

            def fcas_max_power_lower_rule_three(model, time_interval):
                return model.fcas_charge_power[
                           time_interval] >= self.energy_system.energy_storage.charging_power_limit - (
                               model.storage_charge_total[time_interval] + model.storage_discharge_total[
                           time_interval]) - self.bigM * (
                               1 - self.model.fcas_storage_charge_power_is_max[time_interval])

            def fcas_max_power_lower_rule_four(model, time_interval):
                return model.fcas_charge_power[time_interval] >= (
                        self.energy_system.energy_storage.max_capacity - self.model.storage_state_of_charge[
                    time_interval]) / (self.model.eta_chg * fcas_total_duration / minutes_per_hour) - self.bigM * \
                       self.model.fcas_storage_charge_power_is_max[time_interval]

            self.model.fcas_max_power_lower_rule_one_constraint = en.Constraint(self.model.Time,
                                                                                rule=fcas_max_power_lower_rule_one)
            self.model.fcas_max_power_lower_rule_two_constraint = en.Constraint(self.model.Time,
                                                                                rule=fcas_max_power_lower_rule_two)
            # self.model.fcas_max_power_lower_rule_three_constraint = en.Constraint(self.model.Time,
            #                                                                       rule=fcas_max_power_lower_rule_three)
            # self.model.fcas_max_power_lower_rule_four_constraint = en.Constraint(self.model.Time,
            #       


    def update_build_objective(self):
        # Build the objective function ready for optimisation

        if OptimiserObjective.ConnectionPointCost in self.objectives:
            # Connection point cost
            self.objective += sum(self.model.btm_import_tariff[i] * self.model.btm_net_import[i] +  # The cost of purchasing energy
                             self.model.btm_export_tariff[i] * self.model.btm_net_export[i]  # The value of selling energy
                             for i in self.model.Time)

        if OptimiserObjective.ConnectionPointEnergy in self.objectives:
            # The amount of energy crossing the meter boundary
            self.objective += sum((-self.model.btm_net_export[i] + self.model.btm_net_import[i])
                             for i in self.model.Time)

        if OptimiserObjective.GreedyGenerationCharging in self.objectives:
            # Greedy Generation - Favour charging fully from generation in earlier intervals
            self.objective += sum(-self.model.btm_net_export[i]
                             * 1 / self.number_of_intervals
                             * (1 - i / self.number_of_intervals)
                             for i in self.model.Time)

        if OptimiserObjective.GreedyDemandDischarging in self.objectives:
            # Greedy Demand Discharging - Favour satisfying all demand from the storage in earlier intervals
            self.objective += sum(self.model.btm_net_import[i]
                             * 1 / self.number_of_intervals
                             * (1 - i / self.number_of_intervals)
                             for i in self.model.Time)

        if OptimiserObjective.ConnectionPointPeakPower in self.objectives:
            # ToDo - More work is needed to convert this into a demand tariff objective (i.e. a cost etc.)
            self.objective += self.model.peak_connection_point_import_power + self.model.peak_connection_point_export_power


        if OptimiserObjective.ConnectionPointQuantisedPeak in self.objectives:
            # ToDo - What is this objective function? Quantises the Connection point?
            self.objective += sum(self.model.btm_net_export[i] * self.model.btm_net_export[i] +
                             self.model.btm_net_import[i] * self.model.btm_net_import[i]
                             for i in self.model.Time)

        if OptimiserObjective.CapacityAvailability in self.objectives:
            # Implement FCAS style objectives

            # Raise value
            self.objective += sum(self.model.fcas_discharge_power[i] * self.energy_system.capacity_prices.discharge_prices[i] for i in self.model.Time)

            # Lower value
            self.objective += sum(-self.model.fcas_charge_power[i] * self.energy_system.capacity_prices.charge_prices[i] for i in self.model.Time)



        '''if OptimiserObjective.PiecewiseLinear in self.objectives: # ToDo - Fix this implementation to make it complete
            for i in self.energy_system.dispatch.linear_ramp[0]:
                objective += -1 * (self.model.storage_charge_total[i] + self.model.storage_discharge_total[i]) * (
                        1 - self.model.turning_point_two_ramp[i])'''


class LocalEnergyOptimiser(EnergyOptimiser):

    def __init__(self, interval_duration, number_of_intervals, energy_system, objective):
        super().__init__(interval_duration, number_of_intervals, energy_system, objective)

        self.enforce_local_feasability = True
        self.enforce_battery_feasability = True

        self.update_build_model()
        self.update_apply_constraints()
        self.update_build_objective()
        super().optimise()

    def update_build_model(self):
        #### Local Energy Models ####

        # Net import from the grid (without BTM Storage)
        self.model.local_net_import = en.Var(self.model.Time, initialize=self.system_demand_dct)

        # Net export to the grid (without BTM Storage)
        self.model.local_net_export = en.Var(self.model.Time, initialize=self.system_generation_dct)

        # Local consumption (Satisfy local demand from local generation)
        self.model.local_demand_transfer = en.Var(self.model.Time, within=en.NonNegativeReals, initialize=0.0)

        # Local Energy Tariffs
        self.model.le_import_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.le_import_tariff)
        self.model.le_export_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.le_export_tariff)
        self.model.lt_import_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.lt_import_tariff)
        self.model.lt_export_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.lt_export_tariff)
        self.model.re_import_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.re_import_tariff)
        self.model.re_export_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.re_export_tariff)
        self.model.rt_import_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.rt_import_tariff)
        self.model.rt_export_tariff = en.Param(self.model.Time,
                                               initialize=self.energy_system.local_tariff.rt_export_tariff)

        #### Local Grid Flows Peak Power ####

        self.model.local_peak_connection_point_import_power = en.Var(within=en.NonNegativeReals)
        self.model.local_peak_connection_point_export_power = en.Var(within=en.NonNegativeReals)

        def local_peak_connection_point_import(model, interval):
            return model.local_peak_connection_point_import_power >= self.model.storage_charge_grid[interval] + \
                   self.model.storage_discharge_grid[interval] + self.model.local_net_import[interval] + \
                   self.model.local_net_export[interval]

        def local_peak_connection_point_export(model, interval):
            return model.local_peak_connection_point_export_power >= -(self.model.storage_charge_grid[interval] + \
                   self.model.storage_discharge_grid[interval] + self.model.local_net_import[interval] + \
                   self.model.local_net_export[interval])

        self.model.local_peak_connection_point_import_constraint = en.Constraint(self.model.Time,
                                                                           rule=local_peak_connection_point_import)
        self.model.local_peak_connection_point_export_constraint = en.Constraint(self.model.Time,
                                                                           rule=local_peak_connection_point_export)

        """
        TODO For reasons I do not understand, the curtailment export constraint with >= inequality
        causes the solver to fail with a `'c_u_x1636_' is not convex` error. This is only a problem
        with the local optimiser.

        I have not investigated fully, but it may be due to differences in constraints or objectives
        compared to the BTM model.

        The current hack is to enforce equality between system_generation and system_generation_max.
        This has not been tested with an actual `export_limit` applied, presumably this would fail to solve.

        """
        def connection_point_export_curtailment(model, interval):
            """
            Allows generally for generation to be curtailed in order to maximise the objective.
            It is assumed that the supplied system_generation_max
            """
            return model.system_generation[interval] == model.system_generation_max[interval]
        
        self.model.system_generation_curtailment_constraint = en.Constraint(self.model.Time,
            rule=connection_point_export_curtailment
        )                                                                           

    def update_apply_constraints(self):

        # Calculate the customer net energy import
        def local_net_import(model, time_interval):
            return model.local_net_import[time_interval] == model.system_demand[time_interval] + \
                   model.storage_discharge_demand[time_interval] - model.local_demand_transfer[time_interval]

        # calculate the customer net energy export
        def local_net_export(model, time_interval):
            return model.local_net_export[time_interval] == model.system_generation[time_interval] + \
                   model.storage_charge_generation[time_interval] + model.local_demand_transfer[time_interval]

        # constrain the use of local energy exports
        def local_demand_transfer_export(model, time_interval):
            return model.local_demand_transfer[time_interval] + model.storage_charge_generation[time_interval] <= -model.system_generation[time_interval]

        # constrain the use of local energy imports
        def local_demand_transfer_import(model, time_interval):
            return model.storage_discharge_demand[time_interval] - model.local_demand_transfer[time_interval] >= -model.system_demand[time_interval]

        # Add the constraints to the optimisation model
        self.model.local_net_import_constraint = en.Constraint(self.model.Time, rule=local_net_import)
        self.model.local_net_export_constraint = en.Constraint(self.model.Time, rule=local_net_export)
        self.model.local_demand_transfer_export_constraint = en.Constraint(self.model.Time, rule=local_demand_transfer_export)
        self.model.local_demand_transfer_import_constraint = en.Constraint(self.model.Time, rule=local_demand_transfer_import)



        # These set of constraints are designed to enforce the battery to satisfy any residual
        # local demand before discharging to the grid.
        if self.enforce_battery_feasability:
            def electrical_feasability_discharge_grid_one(model: en.ConcreteModel, time_interval: int): # TODO these annotations are probably wrong
                """This constraint (combined with `electrical_feasability_discharge_grid_two`)
                enforces the electrical requirement that the battery must satisfy local demand
                before discharging into the grid. It maps between the boolean variable 
                `local_demand_satisfied` and a bound on `storage_discharge_grid`.

                `local_demand_satisfed = 1` corresponds to a lower bound on `storage_discharge_grid` of zero.
                I.e. if local demand is not satisfied, it is impossible to discharge into the grid

                `local_demand_satisfied = 0` corresponds to a lower bound of `-bigM` (effectively no lower bound).

                Args:
                    model: Pyomo model
                    time_interval: time interval variable

                Returns:
                    obj: constraint object
                """
                return model.storage_discharge_grid[time_interval] >= -self.model.bigM * model.local_demand_satisfied[time_interval]

            def electrical_feasability_discharge_grid_two(model: en.ConcreteModel, time_interval: int):
                """This constraint maps between a boolean `local_demand_satisfied` and its correspondence
                to `storage_discharge_demand`. Combined with `electrical_feasability_discharge_grid_one`,
                this enforces the electrical requirement that the battery must satisfy local demand
                before discharging into the grid.
                
                `local_demand_satisfied = 1` corresponds to `storage_discharge_demand` having the net excess generation
                as an upper bound.

                `local_demand_satisfied = 0` corresponds to `storage_discharge_demand` having an upper bound of 0.

                Args:
                    model: Pyomo model
                    time_interval: time interval passed into constraint equation

                Returns:
                    obj: constraint object
                """
                return model.storage_discharge_demand[time_interval] <= -(model.system_demand[time_interval] + model.system_generation[time_interval]) * model.local_demand_satisfied[time_interval]

            
            self.model.efdc_one = en.Constraint(self.model.Time, rule=electrical_feasability_discharge_grid_one)
            self.model.efdc_two = en.Constraint(self.model.Time, rule=electrical_feasability_discharge_grid_two)

            def electrical_feasability_charge_grid_one(model: en.ConcreteModel, time_interval: int): # TODO these annotations are probably wrong
                """This constraint (combined with `electrical_feasability_charge_grid_two`)
                enforces the electrical requirement that the battery must charge from local
                generation before charging from the grid. It maps between the boolean variable 
                `local_generation_satisfied` and a bound on `storage_charge_grid`.

                `local_generation_satisfied = 1` corresponds to an upper bound on `storage_charge_grid` of `bigM`.
                

                `local_generation_satisfied = 0` corresponds to an upper bound of `0` .
                I.e. if local generation is not accounted for, it is impossible to charge from the grid.

                Args:
                    model: Pyomo model
                    time_interval: time interval variable

                Returns:
                    obj: constraint object
                """
                return model.storage_charge_grid[time_interval] <= self.model.bigM * model.local_generation_satisfied[time_interval]

            def electrical_feasability_charge_grid_two(model: en.ConcreteModel, time_interval: int):
                """This constraint maps between a boolean `local_generation_satisfied` and its correspondence
                to `storage_charge_generation`. Combined with `electrical_feasability_charge_grid_one`,
                this enforces the electrical requirement that the battery must charge from local excess
                generation before charging from the grid.
                
                `local_generation_satisfied = 1` corresponds to `storage_charge_generation` having the net excess generation
                as an upper bound.

                `local_generation_satisfied = 0` corresponds to `storage_charge_generation` having an upper bound of 0.

                Args:
                    model: Pyomo model
                    time_interval: time interval passed into constraint equation

                Returns:
                    obj: constraint object
                """
                return model.storage_charge_generation[time_interval] >= -(model.system_demand[time_interval] + model.system_generation[time_interval]) * model.local_generation_satisfied[time_interval]

            self.model.efcc_one = en.Constraint(self.model.Time, rule=electrical_feasability_charge_grid_one)
            self.model.efcc_two = en.Constraint(self.model.Time, rule=electrical_feasability_charge_grid_two)



        # Additional rules to enforce electrical feasability
        # (Without these rules, the local generation can preferentially export to the grid
        # before satisfying local demand)
        if self.enforce_local_feasability:
            def import_export_rule_one(model: en.ConcreteModel, time_interval: int):
                """Enforce a lower bound on `local_net_export` of `0` or `-bigM` depending on 
                whether `is_local_exporting` is zero or one.

                Args:
                    model (en.ConcreteModel): Pyomo model
                    time_interval (int): time interval passed into constraint

                Returns:
                    obj: constraint object
                """
                return model.local_net_export[time_interval] >= -model.is_local_exporting[time_interval] * self.model.bigM
            
            def import_export_rule_two(model: en.ConcreteModel, time_interval: int):
                """Enforce an upper bound on `local_net_import` of `0` or `bigM` depending on
                whether `is_local_exporting` is one or zero. Combined with `import_export_rule_one`,
                this enforces that the system can only be exporting or importing locally.

                Args:
                    model (en.ConcreteModel): Pyomo model
                    time_interval (int): time interval passed into constraint

                Returns:
                    obj: constraint object
                """
                return model.local_net_import[time_interval] <= (1 - model.is_local_exporting[time_interval]) * self.model.bigM

            self.model.ie_one = en.Constraint(self.model.Time, rule=import_export_rule_one)
            self.model.ie_two = en.Constraint(self.model.Time, rule=import_export_rule_two)

        

    def update_build_objective(self):
        # Build the objective function ready for optimisation

        if OptimiserObjective.LocalModelsCost in self.objectives:
            self.objective += sum((self.model.storage_charge_grid[i] * (self.model.re_import_tariff[i] + self.model.rt_import_tariff[i])) +
                             (self.model.storage_discharge_grid[i] * (self.model.re_export_tariff[i] - self.model.rt_export_tariff[i])) +
                             (self.model.storage_charge_generation[i] * (-self.model.le_export_tariff[i] + self.model.le_import_tariff[i] + self.model.lt_export_tariff[i] + self.model.lt_import_tariff[i])) +
                             (self.model.storage_discharge_demand[i] * (self.model.le_export_tariff[i] - self.model.le_import_tariff[i] - self.model.lt_export_tariff[i] - self.model.lt_import_tariff[i])) +
                             (self.model.local_net_import[i] * (self.model.re_import_tariff[i] + self.model.rt_import_tariff[i])) +
                             (self.model.local_net_export[i] * (self.model.re_export_tariff[i] - self.model.rt_export_tariff[i])) +
                             (self.model.local_demand_transfer[i] * (-self.model.le_export_tariff[i] + self.model.le_import_tariff[i] + self.model.lt_export_tariff[i] + self.model.lt_import_tariff[i]))
                             for i in self.model.Time)

        if OptimiserObjective.LocalThirdParty in self.objectives:
            self.objective += sum((self.model.storage_charge_grid[i] * (self.model.re_import_tariff[i] + self.model.rt_import_tariff[i])) +
                             (self.model.storage_discharge_grid[i] * (self.model.re_export_tariff[i] - self.model.rt_export_tariff[i])) +
                             (self.model.storage_charge_generation[i] * (self.model.le_import_tariff[i] + self.model.lt_import_tariff[i])) +
                             (self.model.storage_discharge_demand[i] * (self.model.le_export_tariff[i] - self.model.lt_export_tariff[i]))
                             for i in self.model.Time)

        if OptimiserObjective.LocalGridPeakPower in self.objectives:
            # ToDo - More work is needed to convert this into a demand tariff objective (i.e. a cost etc.)
            self.objective += self.model.local_peak_connection_point_import_power + self.model.local_peak_connection_point_export_power

        if OptimiserObjective.LocalGridMinimiser in self.objectives:
            # ToDo - What is this objective function? Quantises the Connection point?
            self.objective += sum((self.model.storage_charge_grid[i] + self.model.storage_discharge_grid[i]
                                   + self.model.local_net_import[i] + self.model.local_net_export[i]) *
                                  (self.model.storage_charge_grid[i] + self.model.storage_discharge_grid[i]
                                   + self.model.local_net_import[i] + self.model.local_net_export[i])
                                  for i in self.model.Time) * self.smallM

        if OptimiserObjective.GreedyGenerationCharging in self.objectives:
        #     # Preferentially charge from local solar as soon as possible
        #     # This amounts to minimising the quantity of exported energy in early periods
        #     self.object
            self.objective += sum(self.model.local_net_export[i]
                                * 1 / self.number_of_intervals
                                * (i / self.number_of_intervals)
                                for i in self.model.Time)

        if OptimiserObjective.GreedyDemandDischarging in self.objectives:
            self.objective += sum(self.model.local_net_import[i]
                                * 1 / self.number_of_intervals
                                * (-i / self.number_of_intervals)
                                for i in self.model.Time)
