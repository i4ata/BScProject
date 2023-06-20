# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Regional Integrated model of Climate and the Economy (RICE)
"""
import logging
import os
import sys

from typing import List, Dict, Tuple
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from gym.spaces import MultiDiscrete

_PUBLIC_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [_PUBLIC_REPO_DIR] + sys.path

from rice_helpers import (
    get_abatement_cost,
    get_armington_agg,
    get_aux_m,
    get_capital,
    get_capital_depreciation,
    get_carbon_intensity,
    get_consumption,
    get_damages,
    get_exogenous_emissions,
    get_global_carbon_mass,
    get_global_temperature,
    get_gross_output,
    get_investment,
    get_labor,
    get_land_emissions,
    get_max_potential_exports,
    get_mitigation_cost,
    get_production,
    get_production_factor,
    get_social_welfare,
    get_utility,
    set_rice_params,
)

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_FEATURES = "features"
_ACTION_MASK = "action_mask"
_PROMISES = "promises"
_PROPOSALS = "proposals"


class Rice:
    """
    TODO : write docstring for RICE
    Rice class. Includes all regions, interactions, etc.
    Is initialized based on yaml file.
    etc...
    """

    name = "Rice"

    def __init__(
        self,
        id,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        region_yamls_filename='region_yamls',
        max_negotiation_steps=5,
        message_length=5
    ):
        """TODO : init docstring"""
        self.id = id
        assert (
            num_discrete_action_levels > 1
        ), "the number of action levels should be > 1."
        self.num_discrete_action_levels = num_discrete_action_levels
        self.max_negotiation_steps = max_negotiation_steps
        self.message_length = message_length

        self.float_dtype = np.float32
        self.int_dtype = np.int32

        # Constants
        params, num_regions = set_rice_params(
            #os.path.join(_PUBLIC_REPO_DIR, "region_yamls"),
            os.path.join(_PUBLIC_REPO_DIR, region_yamls_filename),
        )
        # TODO : add to yaml
        self.balance_interest_rate = 0.1

        self.num_regions = num_regions
        self.rice_constant = params["_RICE_CONSTANT"]
        self.dice_constant = params["_DICE_CONSTANT"]
        self.all_constants = self.concatenate_world_and_regional_params(
            self.dice_constant, self.rice_constant
        )

        # TODO: rename constans[0] to dice_constants?
        self.start_year = self.all_constants[0]["xt_0"]
        self.end_year = (
            self.start_year
            + self.all_constants[0]["xDelta"] * self.all_constants[0]["xN"]
        )

        # These will be set in reset (see below)
        self.current_year = None  # current year in the simulation
        self.timestep = None  # episode timestep
        self.activity_timestep = None  # timestep pertaining to the activity stage
        
        self.negotiation_step = None


        # Parameters for Armington aggregation
        # TODO : add to yaml
        self.sub_rate = 0.5
        self.dom_pref = 0.5
        self.for_pref = [0.5 / (self.num_regions - 1)] * self.num_regions

        # Typecasting
        self.sub_rate = np.array([self.sub_rate]).astype(self.float_dtype)
        self.dom_pref = np.array([self.dom_pref]).astype(self.float_dtype)
        self.for_pref = np.array(self.for_pref, dtype=self.float_dtype)

        # Define env global state
        # These will be initialized at reset (see below)
        self.global_state = {}

        # Define the episode length
        self.episode_length = self.dice_constant["xN"]

        # Defining observation and action spaces
        self.observation_space = None  # This will be set via the env_wrapper (in utils)
        #
        self.global_features, self.public_features, self.private_features = self.init_features()
        #


        # Notation nvec: vector of counts of each categorical variable
        # Each region sets mitigation and savings rates
        self.savings_action_nvec = [self.num_discrete_action_levels]
        self.mitigation_rate_action_nvec = [self.num_discrete_action_levels]
        # Each region sets max allowed export from own region
        self.export_action_nvec = [self.num_discrete_action_levels]
        # Each region sets import bids (max desired imports from other countries)
        self.import_actions_nvec = [self.num_discrete_action_levels] * self.num_regions
        # Each region sets import tariffs imposed on other countries
        self.tariff_actions_nvec = [self.num_discrete_action_levels] * self.num_regions

        self.actions_nvec = (
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
        )

        # Set the env action space
        self.action_space = {
            region_id: MultiDiscrete(self.actions_nvec)
            for region_id in range(self.num_regions)
        }

        # Set the default action mask (all ones)
        self.len_actions = sum(self.actions_nvec)
        self.default_agent_action_mask = np.ones(self.len_actions, dtype=bool).reshape(-1, self.num_discrete_action_levels)

    def init_global_negotiation_state(self):
        self.global_negotiation_state = {
            'action_masks' : 
            [
                [
                    [
                        np.copy(self.default_agent_action_mask)
                        for region in range(self.num_regions)
                    ]
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'negotiation_status' : 
            [
                [
                    {
                        sender : {
                            receiver : False
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'promises' : 
            [
                [
                    {
                        sender : {
                            receiver : np.copy(self.default_agent_action_mask)
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'proposals' : 
            [
                [
                    {
                        sender : {
                            receiver : np.copy(self.default_agent_action_mask)
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'decisions' : 
            [
                [
                    {
                        sender : {
                            receiver : np.zeros(1, dtype=self.int_dtype)
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'messages_proposals' : 
            [
                [
                    {
                        sender : {
                            receiver : np.zeros(self.message_length)
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'messages_decisions' : 
            [
                [
                    {
                        sender : {
                            receiver : np.zeros(self.message_length)
                            for receiver in range(self.num_regions)
                            if receiver != sender
                        }
                        for sender in range(self.num_regions)
                    }
                    for negotiation_step in range(self.max_negotiation_steps)
                ]
                for timestep in range(self.episode_length)
            ],

            'rewards_proposals' : np.zeros((self.episode_length, self.max_negotiation_steps, self.num_regions)),
            
            'rewards_decisions' : np.zeros((self.episode_length, self.max_negotiation_steps, self.num_regions))
        }

    def reset(self):
        """
        Reset the environment
        """
        self.timestep = 0
        self.activity_timestep = 0
        self.negotiation_step = 0
        self.current_year = self.start_year

        constants = self.all_constants

        self.set_global_state(
            key="global_temperature",
            value=np.array(
                [constants[0]["xT_AT_0"], constants[0]["xT_LO_0"]],
            ),
            timestep=self.timestep,
            norm=1e1,
        )

        self.set_global_state(
            key="global_carbon_mass",
            value=np.array(
                [
                    constants[0]["xM_AT_0"],
                    constants[0]["xM_UP_0"],
                    constants[0]["xM_LO_0"],
                ],
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="capital_all_regions",
            value=np.array(
                [constants[region_id]["xK_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="labor_all_regions",
            value=np.array(
                [constants[region_id]["xL_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="production_factor_all_regions",
            value=np.array(
                [constants[region_id]["xA_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e2,
        )

        self.set_global_state(
            key="intensity_all_regions",
            value=np.array(
                [
                    constants[region_id]["xsigma_0"]
                    for region_id in range(self.num_regions)
                ]
            ),
            timestep=self.timestep,
            norm=1e-1,
        )

        for key in [
            "global_exogenous_emissions",
            "global_land_emissions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    1,
                ),
                timestep=self.timestep,
            )
        self.set_global_state(
            "timestep", self.timestep, self.timestep, dtype=self.int_dtype, norm=1e2
        )
        self.set_global_state(
            "activity_timestep",
            self.activity_timestep,
            self.timestep,
            dtype=self.int_dtype,
        )

        for key in [
            "capital_depreciation_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "max_export_limit_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    self.num_regions,
                ),
                timestep=self.timestep,
            )

        for key in [
            "consumption_all_regions",
            "current_balance_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "production_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    self.num_regions,
                ),
                timestep=self.timestep,
                norm=1e3,
            )

        for key in [
            "tariffs",
            "future_tariffs",
            "scaled_imports",
            "desired_imports",
            "tariffed_imports",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros((self.num_regions, self.num_regions)),
                timestep=self.timestep,
                norm=1e2,
            )

        self.init_global_negotiation_state()
        return self.generate_observation()

    def init_features(self):
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "consumption_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "max_export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]
        return global_features, public_features, private_features

    def step(self, actions=None):
        """
        The environment step function.
        If negotiation is enabled, it also comprises
        the proposal and evaluation steps.
        """
        # Increment timestep
        self.timestep += 1
        self.negotiation_step = 0

        # Carry over the previous global states to the current timestep
        for key in self.global_state:
            if key != "reward_all_regions":
                self.global_state[key]["value"][self.timestep] = self.global_state[key][
                    "value"
                ][self.timestep - 1].copy()

        self.set_global_state(
            "timestep", self.timestep, self.timestep, dtype=self.int_dtype
        )

        return self.climate_and_economy_simulation_step(actions)

    def register_decisions(self, decisions: Dict[int, np.ndarray]):

        t, n_t, d, n_s = self.timestep, self.negotiation_step, 'decisions', 'negotiation_status'

        # Register decisions into environment
        for agent in range(self.num_regions):

            i = 0
            for other in range(self.num_regions):

                if agent == other or self.global_negotiation_state[n_s][t][max(0, n_t - 1)][agent][other]:
                    continue

                decision = decisions[agent][i].astype(bool)
                self.global_negotiation_state[d][t][n_t][agent][other] = decision

                if decision:
                    self.global_negotiation_state[n_s][t][n_t][agent][other] = True
                    
                    # Update masks
                    # Accept proposal
                    self.global_negotiation_state['action_masks'][t][n_t][agent] = np.logical_and(
                        self.global_negotiation_state['proposals'][t][n_t][other][agent],
                        self.global_negotiation_state['action_masks'][t][max(0, n_t - 1)][agent]
                    )
                    
                    # Keep the promise
                    self.global_negotiation_state['action_masks'][t][n_t][other] = np.logical_and(
                        self.global_negotiation_state['promises'][t][n_t][other][agent],
                        self.global_negotiation_state['action_masks'][t][max(0, n_t - 1)][other]
                    )

                    # If the agent accepts the others' proposal, then the other had a good proposal
                    self.global_negotiation_state['rewards_proposals'][t][n_t][other] += 1
                    self.global_negotiation_state['rewards_decisions'][t][n_t][agent] += 1

                    # Punish the agent if it made a promise that it can't fulfill
                    if np.logical_not(self.global_negotiation_state['action_masks'][t][n_t][agent]).all(0).any():
                        self.global_negotiation_state['rewards_decisions'][t][n_t][agent] -= 10

                i += 1

        # If an agent accepts a proposal from another agent, make sure that the other agent
        # stopped negotiating with it as well
        for agent in range(self.num_regions):
            for other in self.global_negotiation_state[n_s][t][n_t][agent]:
                if self.global_negotiation_state[n_s][t][n_t][agent][other]:
                    self.global_negotiation_state[n_s][t][n_t][other][agent] = True

        # Punish the agents a bit if they never reached agreement
        if n_t == self.max_negotiation_steps - 1:
            for agent in range(self.num_regions):

                # Calculate the number of agents with which the agent did not reach an agreement
                status = self.num_regions - 1 - \
                      sum(self.global_negotiation_state['negotiation_status'][t][n_t][agent].values())

                self.global_negotiation_state['rewards_proposals'][t][n_t][agent] += status
                self.global_negotiation_state['rewards_decisions'][t][n_t][agent] += status

        new_state = self.generate_observation()
        self.negotiation_step += 1
        return new_state

    def register_proposals(self, proposals: Dict[int, Dict[str, np.ndarray]]):

        t, n_t, n_s = self.timestep, self.negotiation_step, 'negotiation_status'
        pp, pm = 'proposals', 'promises'

        if n_t:
            for c in self.global_negotiation_state:
                if not c.startswith('rewards'):
                    self.global_negotiation_state[c][t][n_t] = deepcopy(self.global_negotiation_state[c][t][n_t - 1])

        for agent in range(self.num_regions):
            i = 0
            new_promises = []
            for other in range(self.num_regions):

                if agent == other or self.global_negotiation_state[n_s][t][max(0, n_t - 1)][agent][other]:
                    continue

                # Save proposals and promises in the environment
                proposal = proposals[agent][pp][i].reshape(-1, self.num_discrete_action_levels).astype(bool)
                promise = proposals[agent][pm][i].reshape(-1, self.num_discrete_action_levels).astype(bool)

                self.global_negotiation_state[pp][t][n_t][agent][other] = proposal
                self.global_negotiation_state[pm][t][n_t][agent][other] = promise

                new_promises.append(promise)

                i += 1

            if not new_promises:
                continue
            # Punish the agent if it made promises that it can't keep
            mask = self.global_negotiation_state['action_masks'][t][n_t][agent]
            if np.logical_not(np.logical_and.reduce((*new_promises, mask))).all(0).any():
                self.global_negotiation_state['rewards_proposals'] -= 10

        return self.generate_observation()
                
    def generate_observation(self):
        """
        Generate observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """

        shared_features = np.array([])
        for feature in self.global_features + self.public_features:
            shared_features = np.append(
                shared_features,
                self.flatten_array(
                    self.global_state[feature]["value"][self.timestep]
                    / self.global_state[feature]["norm"]
                ),
            )
        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for region_id in range(self.num_regions):
            # Add a region indicator array to the observation
            region_indicator = np.zeros(self.num_regions, dtype=self.float_dtype)
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in self.private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][self.timestep, region_id]
                        / self.global_state[feature]["norm"]
                    ),
                )

            features_dict[region_id] = all_features

        # Form the observation dictionary keyed by region id.
        t, n_t = max(0, self.timestep - 1), self.negotiation_step
        obs_dict = {}
        for region_id in range(self.num_regions):
            obs_dict[region_id] = {
                _FEATURES: features_dict[region_id],
                _ACTION_MASK: self.global_negotiation_state['action_masks'][t][n_t][region_id],
                _PROMISES: np.stack([
                    self.global_negotiation_state['promises'][t][n_t][sender_id][region_id]
                    for sender_id in range(self.num_regions)
                    if sender_id != region_id
                ]),
                _PROPOSALS: np.stack([
                    self.global_negotiation_state['proposals'][t][n_t][sender_id][region_id]
                    for sender_id in range(self.num_regions)
                    if sender_id != region_id
                ]),
                'negotiation_status':
                    np.array(list(self.global_negotiation_state['negotiation_status'][t][n_t][region_id].values()))
                # 'messages_proposals': {
                #     sender_id: self.global_negotiation_state['messages_proposals'][self.timestep][self.negotiation_step][sender_id][region_id]
                #     for sender_id in range(self.num_regions)
                #     if sender_id != region_id
                # },
                # 'messages_decisions': {
                #     sender_id: self.global_negotiation_state['messages_decisions'][self.timestep][self.negotiation_step][sender_id][region_id]
                #     for sender_id in range(self.num_regions)
                #     if sender_id != region_id
                # }
            }

        return obs_dict

    def climate_and_economy_simulation_step(self, actions=None):
        """
        The step function for the climate and economy simulation.
        PLEASE DO NOT MODIFY THE CODE BELOW.
        These functions dictate the dynamics of the climate and economy simulation,
        and should not be altered hence.
        """
        self.activity_timestep += 1
        self.set_global_state(
            key="activity_timestep",
            value=self.activity_timestep,
            timestep=self.timestep,
            dtype=self.int_dtype,
        )
        assert self.timestep == self.activity_timestep

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        # add actions to global state
        savings_action_index = 0
        mitigation_rate_action_index = savings_action_index + len(
            self.savings_action_nvec
        )
        export_action_index = mitigation_rate_action_index + len(
            self.mitigation_rate_action_nvec
        )
        tariffs_action_index = export_action_index + len(self.export_action_nvec)
        desired_imports_action_index = tariffs_action_index + len(
            self.tariff_actions_nvec
        )

        self.set_global_state(
            "savings_all_regions",
            [
                actions[region_id][savings_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "mitigation_rate_all_regions",
            [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "max_export_limit_all_regions",
            [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "future_tariffs",
            [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "desired_imports",
            [
                actions[region_id][
                    desired_imports_action_index : desired_imports_action_index
                    + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )

        # Constants
        constants = self.all_constants

        const = constants[0]
        aux_m_all_regions = np.zeros(self.num_regions, dtype=self.float_dtype)

        prev_global_temperature = self.get_global_state(
            "global_temperature", self.timestep - 1
        )
        t_at = prev_global_temperature[0]

        # add emissions to global state
        global_exogenous_emissions = get_exogenous_emissions(
            const["xf_0"], const["xf_1"], const["xt_f"], self.activity_timestep
        )
        global_land_emissions = get_land_emissions(
            const["xE_L0"], const["xdelta_EL"], self.activity_timestep, self.num_regions
        )

        self.set_global_state(
            "global_exogenous_emissions", global_exogenous_emissions, self.timestep
        )
        self.set_global_state(
            "global_land_emissions", global_land_emissions, self.timestep
        )
        desired_imports = self.get_global_state("desired_imports")
        scaled_imports = self.get_global_state("scaled_imports")

        for region_id in range(self.num_regions):

            # Actions
            savings = self.get_global_state("savings_all_regions", region_id=region_id)
            mitigation_rate = self.get_global_state(
                "mitigation_rate_all_regions", region_id=region_id
            )

            # feature values from previous timestep
            intensity = self.get_global_state(
                "intensity_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            production_factor = self.get_global_state(
                "production_factor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            capital = self.get_global_state(
                "capital_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            labor = self.get_global_state(
                "labor_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            gov_balance_prev = self.get_global_state(
                "current_balance_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )

            # constants
            const = constants[region_id]

            # climate costs and damages
            mitigation_cost = get_mitigation_cost(
                const["xp_b"],
                const["xtheta_2"],
                const["xdelta_pb"],
                self.activity_timestep,
                intensity,
            )

            damages = get_damages(t_at, const["xa_1"], const["xa_2"], const["xa_3"])
            abatement_cost = get_abatement_cost(
                mitigation_rate, mitigation_cost, const["xtheta_2"]
            )
            production = get_production(
                production_factor,
                capital,
                labor,
                const["xgamma"],
            )

            gross_output = get_gross_output(damages, abatement_cost, production)
            gov_balance_prev = gov_balance_prev * (1 + self.balance_interest_rate)
            investment = get_investment(savings, gross_output)


            for j in range(self.num_regions):
                scaled_imports[region_id][j] = (
                    desired_imports[region_id][j] * gross_output
                )
            # Import bid to self is reset to zero
            scaled_imports[region_id][region_id] = 0

            total_scaled_imports = np.sum(scaled_imports[region_id])
            if total_scaled_imports > gross_output:
                for j in range(self.num_regions):
                    scaled_imports[region_id][j] = (
                        scaled_imports[region_id][j]
                        / total_scaled_imports
                        * gross_output
                    )

            # Scale imports based on gov balance
            init_capital_multiplier = 10.0
            debt_ratio = gov_balance_prev / init_capital_multiplier * const["xK_0"]
            debt_ratio = min(0.0, debt_ratio)
            debt_ratio = max(-1.0, debt_ratio)
            debt_ratio = np.array(debt_ratio).astype(self.float_dtype)
            scaled_imports[region_id] *= 1 + debt_ratio

            self.set_global_state(
                "mitigation_cost_all_regions",
                mitigation_cost,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "damages_all_regions", damages, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "abatement_cost_all_regions",
                abatement_cost,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "production_all_regions", production, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "gross_output_all_regions",
                gross_output,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "current_balance_all_regions",
                gov_balance_prev,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "investment_all_regions",
                investment,
                self.timestep,
                region_id=region_id,
            )

        for region_id in range(self.num_regions):
            x_max = self.get_global_state(
                "max_export_limit_all_regions", region_id=region_id
            )
            gross_output = self.get_global_state(
                "gross_output_all_regions", region_id=region_id
            )
            investment = self.get_global_state(
                "investment_all_regions", region_id=region_id
            )

            # scale desired imports according to max exports
            max_potential_exports = get_max_potential_exports(
                x_max, gross_output, investment
            )
            total_desired_exports = np.sum(scaled_imports[:, region_id])

            if total_desired_exports > max_potential_exports:
                for j in range(self.num_regions):
                    scaled_imports[j][region_id] = (
                        scaled_imports[j][region_id]
                        / total_desired_exports
                        * max_potential_exports
                    )

        self.set_global_state("scaled_imports", scaled_imports, self.timestep)

        # countries with negative gross output cannot import
        prev_tariffs = self.get_global_state(
            "future_tariffs", timestep=self.timestep - 1
        )
        tariffed_imports = self.get_global_state("tariffed_imports")
        scaled_imports = self.get_global_state("scaled_imports")

        for region_id in range(self.num_regions):
            # constants
            const = constants[region_id]

            # get variables from global state
            savings = self.get_global_state("savings_all_regions", region_id=region_id)
            gross_output = self.get_global_state(
                "gross_output_all_regions", region_id=region_id
            )
            investment = get_investment(savings, gross_output)
            labor = self.get_global_state(
                "labor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )

            # calculate tariffed imports, tariff revenue and budget balance
            for j in range(self.num_regions):
                tariffed_imports[region_id, j] = scaled_imports[region_id, j] * (
                    1 - prev_tariffs[region_id, j]
                )
            tariff_revenue = np.sum(
                scaled_imports[region_id, :] * prev_tariffs[region_id, :]
            )

            # Aggregate consumption from domestic and foreign goods
            # domestic consumption
            c_dom = get_consumption(gross_output, investment, exports=scaled_imports[:, region_id])

            consumption = get_armington_agg(
                c_dom=c_dom,
                c_for=tariffed_imports[region_id, :],  # np.array
                sub_rate=self.sub_rate,  # in (0,1)  np.array
                dom_pref=self.dom_pref,  # in [0,1]  np.array
                for_pref=self.for_pref,  # np.array, sums to (1 - dom_pref)
            )

            utility = get_utility(labor, consumption, const["xalpha"])

            social_welfare = get_social_welfare(
                utility, const["xrho"], const["xDelta"], self.activity_timestep
            )

            self.set_global_state(
                "tariff_revenue", tariff_revenue, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "consumption_all_regions",
                consumption,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "utility_all_regions",
                utility,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "social_welfare_all_regions",
                social_welfare,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "reward_all_regions",
                utility,
                self.timestep,
                region_id=region_id,
            )

        # Update gov balance
        for region_id in range(self.num_regions):
            const = constants[region_id]
            gov_balance_prev = self.get_global_state(
                "current_balance_all_regions", region_id=region_id
            )
            scaled_imports = self.get_global_state("scaled_imports")

            gov_balance = gov_balance_prev + const["xDelta"] * (
                np.sum(scaled_imports[:, region_id])
                - np.sum(scaled_imports[region_id, :])
            )
            self.set_global_state(
                "current_balance_all_regions",
                gov_balance,
                self.timestep,
                region_id=region_id,
            )

        self.set_global_state(
            "tariffed_imports",
            tariffed_imports,
            self.timestep,
        )

        # Update temperature
        m_at = self.get_global_state("global_carbon_mass", timestep=self.timestep - 1)[
            0
        ]
        prev_global_temperature = self.get_global_state(
            "global_temperature", timestep=self.timestep - 1
        )

        global_exogenous_emissions = self.get_global_state(
            "global_exogenous_emissions"
        )[0]

        const = constants[0]
        global_temperature = get_global_temperature(
            np.array(const["xPhi_T"]),
            prev_global_temperature,
            const["xB_T"],
            const["xF_2x"],
            m_at,
            const["xM_AT_1750"],
            global_exogenous_emissions,
        )
        self.set_global_state(
            "global_temperature",
            global_temperature,
            self.timestep,
        )

        for region_id in range(self.num_regions):
            intensity = self.get_global_state(
                "intensity_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            mitigation_rate = self.get_global_state(
                "mitigation_rate_all_regions", region_id=region_id
            )
            production = self.get_global_state(
                "production_all_regions", region_id=region_id
            )
            land_emissions = self.get_global_state("global_land_emissions")

            aux_m = get_aux_m(
                intensity,
                mitigation_rate,
                production,
                land_emissions,
            )
            aux_m_all_regions[region_id] = aux_m

        # Update carbon mass
        const = constants[0]
        prev_global_carbon_mass = self.get_global_state(
            "global_carbon_mass", timestep=self.timestep - 1
        )
        global_carbon_mass = get_global_carbon_mass(
            const["xPhi_M"],
            prev_global_carbon_mass,
            const["xB_M"],
            np.sum(aux_m_all_regions),
        )
        self.set_global_state("global_carbon_mass", global_carbon_mass, self.timestep)

        for region_id in range(self.num_regions):
            capital = self.get_global_state(
                "capital_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            labor = self.get_global_state(
                "labor_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            production_factor = self.get_global_state(
                "production_factor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            intensity = self.get_global_state(
                "intensity_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            investment = self.get_global_state(
                "investment_all_regions", timestep=self.timestep, region_id=region_id
            )

            const = constants[region_id]

            capital_depreciation = get_capital_depreciation(
                const["xdelta_K"], const["xDelta"]
            )
            updated_capital = get_capital(
                capital_depreciation, capital, const["xDelta"], investment
            )
            updated_capital = updated_capital

            updated_labor = get_labor(labor, const["xL_a"], const["xl_g"])
            updated_production_factor = get_production_factor(
                production_factor,
                const["xg_A"],
                const["xdelta_A"],
                const["xDelta"],
                self.activity_timestep,
            )
            updated_intensity = get_carbon_intensity(
                intensity,
                const["xg_sigma"],
                const["xdelta_sigma"],
                const["xDelta"],
                self.activity_timestep,
            )

            self.set_global_state(
                "capital_depreciation_all_regions",
                capital_depreciation,
                self.timestep,
            )
            self.set_global_state(
                "capital_all_regions",
                updated_capital,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "labor_all_regions",
                updated_labor,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "production_factor_all_regions",
                updated_production_factor,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "intensity_all_regions",
                updated_intensity,
                self.timestep,
                region_id=region_id,
            )

        self.set_global_state(
            "tariffs",
            self.global_state["future_tariffs"]["value"][self.timestep],
            self.timestep,
        )

        obs = self.generate_observation()
        rew = {
            region_id: self.global_state["reward_all_regions"]["value"][
                self.timestep, region_id
            ]
            for region_id in range(self.num_regions)
        }
        # Set current year
        self.current_year += self.all_constants[0]["xDelta"]
        done = {"__all__": self.current_year == self.end_year}
        info = {}
        return obs, rew, done, info

    def estimate_reward_distribution(self, n_trials: int = 1_000):
        rewards: np.ndarray = np.zeros((self.episode_length * n_trials, self.num_regions))
        for trial in tqdm(range(n_trials)):
            state = self.reset()
            for step in range(self.episode_length):
                
                # Step in the environment with a random action
                state, reward, _, _ = self.step({
                    agent_id : np.random.randint(self.num_discrete_action_levels, size=len(self.action_space[agent_id]))
                    for agent_id in range(self.num_regions)
                })

                # Save the reward
                rewards[trial * self.episode_length + step] = list(reward.values())
        
        self.mean_random_reward = rewards.mean(axis = 1)
        return self.mean_random_reward

    def set_global_state(
        self, key=None, value=None, timestep=None, norm=None, region_id=None, dtype=None
    ):
        """
        Set a specific slice of the environment global state with a key and value pair.
        The value is set for a specific timestep, and optionally, a specific region_id.
        Optionally, a normalization factor (used for generating observation),
        and a datatype may also be provided.
        """
        assert key is not None
        assert value is not None
        assert timestep is not None
        if norm is None:
            norm = 1.0
        if dtype is None:
            dtype = self.float_dtype

        if isinstance(value, list):
            value = np.array(value, dtype=dtype)
        elif isinstance(value, (float, np.floating)):
            value = np.array([value], dtype=self.float_dtype)
        elif isinstance(value, (int, np.integer)):
            value = np.array([value], dtype=self.int_dtype)
        else:
            assert isinstance(value, np.ndarray)

        if key not in self.global_state:
            logging.info("Adding %s to global state.", key)
            if region_id is None:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,) + value.shape, dtype=dtype
                    ),
                    "norm": norm,
                }
            else:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,) + (self.num_regions,) + value.shape,
                        dtype=dtype,
                    ),
                    "norm": norm,
                }

        # Set the value
        if region_id is None:
            self.global_state[key]["value"][timestep] = value
        else:
            self.global_state[key]["value"][timestep, region_id] = value

    def get_global_state(self, key=None, timestep=None, region_id=None):
        assert key in self.global_state, f"Invalid key '{key}' in global state!"
        if timestep is None:
            timestep = self.timestep
        if region_id is None:
            return self.global_state[key]["value"][timestep].copy()
        return self.global_state[key]["value"][timestep, region_id].copy()

    @staticmethod
    def flatten_array(array):
        """Flatten a numpy array"""
        return np.reshape(array, -1)

    def concatenate_world_and_regional_params(self, world, regional):
        """
        This function merges the world params dict into the regional params dict.
        Inputs:
            world: global params, dict, each value is common to all regions.
            regional: region-specific params, dict,
                      length of the values should equal the num of regions.
        Outputs:
            outs: list of dicts, each dict corresponding to a region
                  and comprises the global and region-specific parameters.
        """
        vals = regional.values()
        assert all(
            len(item) == self.num_regions for item in vals
        ), "The number of regions has to be consistent!"

        outs = []
        for region_id in range(self.num_regions):
            out = world.copy()
            for key, val in regional.items():
                out[key] = val[region_id]
            outs.append(out)
        return outs