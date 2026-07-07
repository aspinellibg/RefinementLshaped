import sys
import re
import time
import copy

import pandas as pd
import os
import re

import networkx as nx
import matplotlib.pyplot as plt

import random

from SMNDP_model import *

from gurobipy import GRB

class Node:

    def __init__(self,i):
        self.Index=i
        self.outboundArcs={}
        self.inboundArcs={}
        self.outboundCommodities={}
        self.inboundCommodities={}

    def __repr__(self):
        return str(self.Index)

class Arc:

    def __init__(self,i):
        self.Index=i
        self.Origin=None
        self.Destination=None
        self.FixedCost=None
        self.VariableCost=None
        self.Capacity=None #this will be defined in the scenario

    def __repr__(self):
        return str(self.Origin) + "->" + str(self.Destination) + " @FC " + str(self.FixedCost) + " @VC " + str(self.VariableCost)

class Commodity:
    
    def __init__(self,i):
        self.Index=i
        self.Origin=None
        self.Destination=None
        self.Size=None #this will be defined in the scenario

    def __repr__(self):
        return str(self.Origin) + "->" + str(self.Destination) 

class Scenario:

    def __init__(self,i):
        self.Index=i
        self.Probability=0
        self.CommodityDemand={}
        self.ArcCapacity={}

    def saveCommodityDemand(self,c,d):
        self.CommodityDemands[c]=d

    def saveArcCapacity(self,a,d):
        self.ArcCapacity[a]=d

class Instance:

    def __init__(self):
        self.Nodes={}
        self.Arcs={}
        self.Commodities={}
        self.Scenarios={}
        self.DummyArcs = {}


    def getNumberArcs(self):
        return len(self.Arcs.keys())

    def getNumberCommodities(self):
        return len(self.Commodities.keys())

    def saveArc(self,i,o,d,vc,c,fc):
        if o not in self.Nodes:
            self.Nodes[o]=Node(o)
        if d not in self.Nodes:
            self.Nodes[d]=Node(d)
        a=Arc(i)
        a.Origin=self.Nodes[o]
        a.Destination=self.Nodes[d]
        a.VariableCost=vc
        a.FixedCost=fc
        a.Origin.outboundArcs[a.Destination]=a
        a.Destination.inboundArcs[a.Origin]=a
        self.Arcs[i]=a

    def saveCommodity(self,i,o,d,dmd):
        cmd=Commodity(i)
        cmd.Origin=self.Nodes[o]
        cmd.Destination=self.Nodes[d]
        cmd.Demand=dmd 
        cmd.Origin.outboundCommodities[d]=cmd
        cmd.Destination.inboundCommodities[o]=cmd
        self.Commodities[i]=cmd

    def saveScenario(self,i,scen):
        self.Scenarios[i]=scen
        
    def saveDummyArc(self, index, origin, destination, fixed_cost=0.0, var_cost=1e6, cap=float('inf')):
        arc = Arc(index)
        arc.Origin = self.Nodes[origin]
        arc.Destination = self.Nodes[destination]
        arc.FixedCost = fixed_cost
        arc.VariableCost = var_cost
        arc.Capacity = cap
        self.DummyArcs[index] = arc


def readNetworkFile(fname):

    f=open(fname,'r')
    lns=f.readlines()[1:]
    nln=re.sub(r"\s+", ",",lns[0])
    cols=nln[1:].split(',')
    nNodes=int(cols[0])
    nArcs=int(cols[1])
    nCommods=int(cols[2])
    inst=Instance()

    for i in range(0,nArcs):
        ln=lns[1+i]
        nln=re.sub(r"\s+", ",",ln)
        cols=nln[1:].split(',')
        origin=int(cols[0])
        dest=int(cols[1])
        varCost=float(cols[2])
        cap=float(cols[3])
        fixedCost=float(cols[4])
        inst.saveArc(i,origin,dest,varCost,cap,fixedCost)

    for i in range(0,nCommods):
        ln=lns[1+nArcs+i]
        nln=re.sub(r"\s+", ",",ln)
        cols=nln[1:].split(',')
        origin=int(cols[0])
        dest=int(cols[1])
        demand=float(cols[2])
        inst.saveCommodity(i,origin,dest,demand)

    return inst


def readScenarioFile(fname,inst):
    f=open(fname,'r')
    lns=f.readlines()
    nScens=int(lns[0])
    print("Number of scenarios:", nScens)
    cnt=0
    nArcs=inst.getNumberArcs()
    nCommods=inst.getNumberCommodities()
    for ln in lns[1:]:
        nln=re.sub(r"\s+", ",",ln)
        scen=Scenario(cnt)
        cols=nln.split(',')
        prb=float(cols[0])
        scen.Probability=prb
        for i in range(0,nArcs):
            cap=float(cols[1+i])
            scen.ArcCapacity[i]=cap
        for i in range(0,nCommods):
            dmd=float(cols[1+nArcs+i])
            scen.CommodityDemand[i]=dmd
        inst.saveScenario(cnt,scen)
        cnt = cnt+1 


def create_grouped_instance(original_instance, group_size):
    all_scenarios = list(original_instance.Scenarios.items())
    num_scenarios = len(all_scenarios)
    grouped_scenarios = []
    intralevel_weights = []
    scenario_group_map = {}

    group_id = 0
    for i in range(0, num_scenarios, group_size):
        group = all_scenarios[i:i+group_size]
        total_prob = sum(s.Probability for _, s in group)
        group_info = []
        for idx, (sid, scenario) in enumerate(group):
            scenario_group_map[sid] = group_id
            group_info.append((sid, scenario.Probability / total_prob))  # normalized
        grouped_scenarios.append(group_info)
        intralevel_weights.append(total_prob)
        group_id += 1

    return grouped_scenarios, intralevel_weights, scenario_group_map


def create_grouped_instance_fixedscenarios(original_instance, group_size):
    all_scenarios = list(original_instance.Scenarios.items())
    num_scenarios = len(all_scenarios)

    # Fixed scenario = first scenario of the scenario tree
    fixed_sid, fixed_scenario = all_scenarios[0]
    fixed_prob = fixed_scenario.Probability

    # Remaining scenarios
    nonfixed_scenarios = all_scenarios[1:]
    num_nonfixed = len(nonfixed_scenarios)

    nonfixed_per_group = group_size - 1

    if num_nonfixed % nonfixed_per_group != 0:
        raise ValueError(
            f"Number of non-fixed scenarios ({num_nonfixed}) must be divisible by "
            f"group_size - 1 ({nonfixed_per_group})."
        )

    denom_nonfixed = 1.0 - fixed_prob
    if abs(denom_nonfixed) < 1e-12:
        raise ValueError(
            "1 - probability of the fixed scenario is zero; cannot compute intralevel weights."
        )

    grouped_scenarios = []
    intralevel_weights = []
    scenario_group_map = {}

    group_id = 0
    for i in range(0, num_nonfixed, nonfixed_per_group):
        nonfixed_group = nonfixed_scenarios[i:i + nonfixed_per_group]

        sum_nonfixed_group = sum(s.Probability for _, s in nonfixed_group)
        intralevel_weight = sum_nonfixed_group / denom_nonfixed

        group_info = []

        # Fixed scenario
        group_info.append((fixed_sid, fixed_prob))

        # Non-fixed scenarios
        for sid, scenario in nonfixed_group:
            updated_prob = scenario.Probability / intralevel_weight
            group_info.append((sid, updated_prob))
            scenario_group_map[sid] = group_id

        grouped_scenarios.append(group_info)
        intralevel_weights.append(intralevel_weight)

        group_id += 1

    # fixed scenario belongs to all groups
    scenario_group_map[fixed_sid] = list(range(group_id))

    print("\nGenerated Groups (Fixed Scenario =", fixed_sid, ")\n")

    for g, group in enumerate(grouped_scenarios):
        scenario_ids = []
        probs = []

        for sid, prob in group:
            scenario_ids.append(str(sid))
            probs.append(f"{prob:.6f}")

        scen_str = "(" + " ".join(scenario_ids) + ")"
        prob_str = "(" + " ".join(probs) + ")"
        weight = intralevel_weights[g]

        print(f"{scen_str}, {prob_str}, weight = {weight:.6f}")

    print()

    return grouped_scenarios, intralevel_weights, scenario_group_map


inst=readNetworkFile(sys.argv[1])
readScenarioFile(sys.argv[2],inst)
typeofmodel = sys.argv[3]


for i, k in enumerate(inst.Commodities.values()):
    inst.saveDummyArc(
        index=i,
        origin=k.Origin.Index,
        destination=k.Destination.Index,
        fixed_cost=0.0,
        var_cost=2e2,
        cap=float('inf')
    )


for arc_id, arc in inst.DummyArcs.items():
    print(f"Dummy Arc {arc_id}: {arc.Origin.Index} -> {arc.Destination.Index}, "
          f"FixedCost={arc.FixedCost}, VarCost={arc.VariableCost}, Capacity={arc.Capacity}")



# Print the main features of the network
print("Number of nodes:", len(inst.Nodes)) # Number of nodes
print(inst.Nodes) # Set of nodes
print("Number of arcs:", len(inst.Arcs)) # Number of arcs
print(inst.Arcs) # Set of arcs
print("Number of commodities:", len(inst.Commodities)) # Number of commodities
print(inst.Commodities) # Set of commodities


time_limit_model = 86400 # time limit

alpha = 0.95 # confidence level CVaR
beta = 0.5 # risk-aversion weight: (1-beta) * expectation + beta * CVaR

costdummy = 2e2

time_model_start = time.time()


if typeofmodel == 'bendersGrouping_disjoint':
    group_size = int(sys.argv[4])  # Number of scenarios per group

    # Create group structure from instance
    grouped_scenarios, intralevel_weights, scenario_group_map = create_grouped_instance(inst, group_size)
    
    # Solve grouped Benders
    time_model_start = time.time()
    model_bendersGrouping_disjoint, x_vars, solTau, LB_list, UB_list, abs_gap_list, rel_gap_list, num_opt_cut = solveBenders_grouped(
        inst, grouped_scenarios, intralevel_weights, alpha, beta, costdummy, time_limit_model
    )
    time_model_end = time.time()
    time_model = min(time_limit_model, time_model_end - time_model_start)

    print("Time Benders Grouping (with disjoint group) (s):", time_model)

    print("\nSelected arcs:")
    for a in inst.Arcs.values():
        if x_vars.get(a.Index, 0) + 0.5 > 1:
            print(f"  Arc {a.Index}: {a.Origin.Index} -> {a.Destination.Index}, FixedCost = {a.FixedCost}")

    first_stage_cost = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())
    print(f"\nFirst-stage (design) cost: {first_stage_cost:.4f}")

    print(f"Value at Risk (VaR) τ: = {solTau.X:.4f}")

    print(f"Total objective value: {model_bendersGrouping_disjoint.ObjVal:.4f}")

    print(f"Number of optimality cuts: {num_opt_cut:.1f}")

    log_LB = []
    log_UB = []
    log_rel_gap = []

    for lb, ub in zip(LB_list, UB_list):
        if lb > 0 and ub > 0:
            log_LB.append(np.log10(lb))
            log_UB.append(np.log10(ub))
        else:
            log_LB.append(np.nan)
            log_UB.append(np.nan)
    for rel in rel_gap_list:
        log_rel_gap.append(np.log10(rel))
    df_benders_progress = pd.DataFrame({
        "Iteration": range(1, len(LB_list) + 1),
        'LB': LB_list,
        'UB': UB_list,
        'AbsGap': abs_gap_list,
        'RelGap': rel_gap_list,
        'log(LB)': log_LB,
        'log(UB)': log_UB,
        'log(RelGap)': log_rel_gap
    })


    netfile = sys.argv[1]
    scenfile = sys.argv[2]

    scen_norm = os.path.normpath(scenfile)
    scenario_set = os.path.basename(os.path.dirname(scen_norm))

    scenario_file = os.path.splitext(os.path.basename(scen_norm))[0]

    output_dir = os.path.join("Results", "LB_UB_grouped")
    os.makedirs(output_dir, exist_ok=True)

    progress_csv_path = os.path.join(
        output_dir,
        f"LB_UB_groupsize_"
        f"{group_size}_"
        f"disjoint_"
        f"{scenario_set}_"
        f"{scenario_file}_"
        f"sequentialgrouping.csv"
    )

    df_benders_progress.to_csv(progress_csv_path, index=False)
    print(f"\nBenders progress saved to '{progress_csv_path}'")

    scenfile_norm = os.path.normpath(scenfile)
    scenario_file = os.path.basename(scenfile_norm)
    scenario_set = os.path.basename(os.path.dirname(scenfile_norm)) 
    match = re.search(r'scens_(\d+)_', scenfile)
    num_scenarios = int(match.group(1)) if match else len(inst.Scenarios)

    last_lower_bound = LB_list[-1] if LB_list else float('-inf')
    num_arcs_benders = sum(1 for val in x_vars.values() if val > 0.5)
    first_stage_benders = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())
    tau_benders_val = solTau.X

    row = {
    "network": os.path.basename(netfile),
    "scenario_set": scenario_set,
    "scenario_file": scenario_file,
    "num_scenarios": num_scenarios,
    "group_size": group_size,
    "num_subgroups": len(intralevel_weights),
    "benders_lastLB": LB_list[-1],
    "benders_lastUB": UB_list[-1],
    "benders_lastRelGap": rel_gap_list[-1],
    "benders_first_stage": first_stage_benders,
    "benders_tau": tau_benders_val,
    "benders_time": time_model,
    "benders_iterations": len(LB_list),
    "benders_num_opt_cut": num_opt_cut
    }

    results_csv_path = "Results_bendersGrouping_disjoint.csv"

    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row])

    df_updated.to_csv(results_csv_path, index=False)
    print(f"\nResults appended to '{results_csv_path}'")


elif typeofmodel == 'bendersGrouping_fixedscenarios':
    group_size = int(sys.argv[4])  # Number of scenarios per group

    # Create group structure from instance
    grouped_scenarios, intralevel_weights, scenario_group_map = create_grouped_instance_fixedscenarios(inst, group_size)

    # Solve grouped Benders
    time_model_start = time.time()
    model_bendersGrouping_fixedscenarios, x_vars, solTau, LB_list, UB_list, abs_gap_list, rel_gap_list, num_opt_cut = solveBenders_grouped(
        inst, grouped_scenarios, intralevel_weights, alpha, beta, costdummy, time_limit_model
    )
    time_model_end = time.time()
    time_model = min(time_limit_model, time_model_end - time_model_start)

    print("Time Benders Grouping (with fixed scenarios group) (s):", time_model)

    print("\nSelected arcs:")
    for a in inst.Arcs.values():
        if x_vars.get(a.Index, 0) + 0.5 > 1:
            print(f"  Arc {a.Index}: {a.Origin.Index} -> {a.Destination.Index}, FixedCost = {a.FixedCost}")

    first_stage_cost = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())
    print(f"\nFirst-stage (design) cost: {first_stage_cost:.4f}")

    print(f"Value at Risk (VaR) τ: = {solTau.X:.4f}")

    print(f"Total objective value: {model_bendersGrouping_fixedscenarios.ObjVal:.4f}")

    print(f"Number of optimality cuts: {num_opt_cut:.1f}")

    log_LB = []
    log_UB = []
    log_rel_gap = []

    for lb, ub in zip(LB_list, UB_list):
        if lb > 0 and ub > 0:
            log_LB.append(np.log10(lb))
            log_UB.append(np.log10(ub))
        else:
            log_LB.append(np.nan)
            log_UB.append(np.nan)
    for rel in rel_gap_list:
        log_rel_gap.append(np.log10(rel))
    df_benders_progress = pd.DataFrame({
        "Iteration": range(1, len(LB_list) + 1),
        'LB': LB_list,
        'UB': UB_list,
        'AbsGap': abs_gap_list,
        'RelGap': rel_gap_list,
        'log(LB)': log_LB,
        'log(UB)': log_UB,
        'log(RelGap)': log_rel_gap
    })

    netfile = sys.argv[1]
    scenfile = sys.argv[2]

    scen_norm = os.path.normpath(scenfile)
    scenario_set = os.path.basename(os.path.dirname(scen_norm))
    scenario_file = os.path.splitext(os.path.basename(scen_norm))[0]
    output_dir = os.path.join("Results", "LB_UB_grouped")
    os.makedirs(output_dir, exist_ok=True)

    progress_csv_path = os.path.join(
        output_dir,
        f"LB_UB_groupsize_"
        f"{group_size}_"
        f"fixed_scenarios_"
        f"{scenario_set}_"
        f"{scenario_file}.csv"
    )

    df_benders_progress.to_csv(progress_csv_path, index=False)
    print(f"\nBenders progress saved to '{progress_csv_path}'")

    scenfile_norm = os.path.normpath(scenfile)
    scenario_file = os.path.basename(scenfile_norm)
    scenario_set = os.path.basename(os.path.dirname(scenfile_norm)) 

    match = re.search(r'scens_(\d+)_', scenfile)
    num_scenarios = int(match.group(1)) if match else len(inst.Scenarios)

    last_lower_bound = LB_list[-1] if LB_list else float('-inf')
    num_arcs_benders = sum(1 for val in x_vars.values() if val > 0.5)
    first_stage_benders = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())

    tau_benders_val = solTau.X


    row = {
    "network": os.path.basename(netfile),
    "scenario_set": scenario_set,
    "scenario_file": scenario_file,
    "num_scenarios": num_scenarios,
    "group_size": group_size,
    "num_subgroups": len(intralevel_weights),
    "benders_lastLB": LB_list[-1],
    "benders_lastUB": UB_list[-1],
    "benders_lastRelGap": rel_gap_list[-1],
    "benders_first_stage": first_stage_benders,
    "benders_tau": tau_benders_val,
    "benders_time": time_model,
    "benders_iterations": len(LB_list),
    "benders_num_opt_cut": num_opt_cut
    }

    results_csv_path = "Results_bendersGrouping_fixedscenarios.csv"

    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row])

    df_updated.to_csv(results_csv_path, index=False)
    print(f"\nResults appended to '{results_csv_path}'")




elif typeofmodel == 'bendersGrouping_disjoint_ALGO':
    group_size_lower_level = int(sys.argv[4])  # Number of scenarios per group at level j-1
    group_size_upper_level = int(sys.argv[5])  # Number of scenarios per group at level j

    # Create group structures from instance
    grouped_scenarios_lower_level, intralevel_weights_lower_level, scenario_group_map_lower_level = create_grouped_instance(inst, group_size_lower_level)
    grouped_scenarios_upper_level, intralevel_weights_upper_level, scenario_group_map_upper_level = create_grouped_instance(inst, group_size_upper_level)


    interlevel_weights = []

    for idx_lower, lower_group in enumerate(grouped_scenarios_lower_level):
        first_scenario_id = lower_group[0][0]

        upper_group_id = scenario_group_map_upper_level[first_scenario_id]

        weight_lower = intralevel_weights_lower_level[idx_lower]
        weight_upper = intralevel_weights_upper_level[upper_group_id]

        interlevel_weight = weight_lower / weight_upper
    
        interlevel_weights.append(interlevel_weight)


    # Solve grouped Benders
    time_model_start = time.time()
    model_bendersGrouping_disjoint, x_vars, solTau, LB_list, UB_list, abs_gap_list, rel_gap_list, num_opt_cut = solveBenders_grouped_algo(
        inst, grouped_scenarios_lower_level, intralevel_weights_lower_level,
        grouped_scenarios_upper_level, intralevel_weights_upper_level,
        interlevel_weights,
        alpha, beta, costdummy, time_limit_model,
        scenario_group_map_upper_level
    )
    time_model_end = time.time()
    time_model = min(time_limit_model, time_model_end - time_model_start)

    print("Time Benders Grouping (with disjoint group and algorithm) (s):", time_model)

    print("\nSelected arcs:")
    for a in inst.Arcs.values():
        if x_vars.get(a.Index, 0) + 0.5 > 1:
            print(f"  Arc {a.Index}: {a.Origin.Index} -> {a.Destination.Index}, FixedCost = {a.FixedCost}")

    first_stage_cost = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())
    print(f"\nFirst-stage (design) cost: {first_stage_cost:.4f}")

    print(f"Value at Risk (VaR) τ: = {solTau.X:.4f}")

    print(f"Total objective value: {model_bendersGrouping_disjoint.ObjVal:.4f}")

    print(f"Number of optimality cuts: {num_opt_cut:.1f}")

    log_LB = []
    log_UB = []
    log_rel_gap = []

    for lb, ub in zip(LB_list, UB_list):
        if lb > 0 and ub > 0:
            log_LB.append(np.log10(lb))
            log_UB.append(np.log10(ub))
        else:
            log_LB.append(np.nan)
            log_UB.append(np.nan)
    for rel in rel_gap_list:
        log_rel_gap.append(np.log10(rel))
    df_benders_progress = pd.DataFrame({
        "Iteration": range(1, len(LB_list) + 1),
        'LB': LB_list,
        'UB': UB_list,
        'AbsGap': abs_gap_list,
        'RelGap': rel_gap_list,
        'log(LB)': log_LB,
        'log(UB)': log_UB,
        'log(RelGap)': log_rel_gap
    })

    netfile = sys.argv[1]
    scenfile = sys.argv[2]

    scen_norm = os.path.normpath(scenfile)
    scenario_set = os.path.basename(os.path.dirname(scen_norm))

    scenario_file = os.path.splitext(os.path.basename(scen_norm))[0]

    output_dir = os.path.join("Results", "LB_UB_grouped_ALGO")
    os.makedirs(output_dir, exist_ok=True)

    progress_csv_path = os.path.join(
        output_dir,
        f"LB_UB_groupsize_"
        f"{group_size_upper_level}_"
        f"disjoint_"
        f"{scenario_set}_"
        f"{scenario_file}_"
        f"ALGO.csv"
    )

    df_benders_progress.to_csv(progress_csv_path, index=False)
    print(f"\nBenders progress saved to '{progress_csv_path}'")


    scenfile_norm = os.path.normpath(scenfile)
    scenario_file = os.path.basename(scenfile_norm)
    scenario_set = os.path.basename(os.path.dirname(scenfile_norm)) 

    match = re.search(r'scens_(\d+)_', scenfile)
    num_scenarios = int(match.group(1)) if match else len(inst.Scenarios)

    last_lower_bound = LB_list[-1] if LB_list else float('-inf')
    
    num_arcs_benders = sum(1 for val in x_vars.values() if val > 0.5)
    first_stage_benders = sum(a.FixedCost * x_vars[a.Index] for a in inst.Arcs.values())
    tau_benders_val = solTau.X

    row = {
    "network": os.path.basename(netfile),
    "scenario_set": scenario_set,
    "scenario_file": scenario_file,
    "num_scenarios": num_scenarios,
    "group_size_upper_level": group_size_upper_level,
    "num_subgroups": len(intralevel_weights_upper_level),
    "benders_lastLB": LB_list[-1],
    "benders_lastUB": UB_list[-1],
    "benders_lastRelGap": rel_gap_list[-1],
    "benders_first_stage": first_stage_benders,
    "benders_tau": tau_benders_val,
    "benders_time": time_model,
    "benders_iterations": len(LB_list),
    "benders_num_opt_cut": num_opt_cut
    }


    results_csv_path = "Results_bendersGrouping_disjoint_ALGO.csv"

    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row])

    df_updated.to_csv(results_csv_path, index=False)
    print(f"\nResults appended to '{results_csv_path}'")


else:
    raise Exception(f"Choose among bendersGrouping_disjoint, bendersGrouping_fixedscenarios, bendersGrouping_disjoint_ALGO")











