import numpy as np
import gurobipy as gp
from gurobipy import *
import time
from collections import defaultdict
from gurobipy import LinExpr


def master_problem_grouped(inst, num_groups, intralevel_weights, alpha, beta):
    design_arcs = list(inst.Arcs.values())

    master = Model("Benders_Master_Grouped")
    master.setParam("OutputFlag", 0)
    master.setParam('MIPGap', 1e-5) 

    x = master.addVars([a.Index for a in design_arcs], vtype=GRB.BINARY, name="x")
    
    tau = master.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="tau")
    theta = master.addVars(range(num_groups), lb=0.0, vtype=GRB.CONTINUOUS, name="theta")

    master.setObjective(
        quicksum(a.FixedCost * x[a.Index] for a in design_arcs) +
        beta * tau +
        quicksum(intralevel_weights[g] * theta[g] for g in range(num_groups)),
        GRB.MINIMIZE
    )

    return master, x, tau, theta


def dual_subproblem_grouped(inst, group, x_vals, tau_val, alpha, beta, costdummy):
    model = Model("Dual_Group_Subproblem")
    model.setParam("OutputFlag", 0)
    model.setParam('MIPGap', 1e-5)

    design_arcs = list(inst.Arcs.values())
    dummy_arcs = list(inst.DummyArcs.values())
    commodities = list(inst.Commodities.values())
    nodes = list(inst.Nodes.values())

    lambda_vars = model.addVars([(sid, k.Index, n.Index)
                                  for sid, _ in group
                                  for k in commodities for n in nodes],
                                 lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambda")

    mu_vars = model.addVars([(sid, a.Index)
                             for sid, _ in group
                             for a in design_arcs],
                            lb=0.0, vtype=GRB.CONTINUOUS, name="mu")

    gamma_vars = model.addVars([sid for sid, _ in group], lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")

    obj = 0.0
    for sid, _ in group:
        s = inst.Scenarios[sid]

        v_expr = quicksum(
            s.CommodityDemand[k.Index] * (
                lambda_vars[sid, k.Index, k.Origin.Index] - lambda_vars[sid, k.Index, k.Destination.Index]
            ) for k in commodities
        )
        cap_expr = quicksum(
            s.ArcCapacity[a.Index] * x_vals[a.Index] * mu_vars[sid, a.Index] for a in design_arcs
        )
        obj += v_expr - cap_expr - tau_val * gamma_vars[sid]

    model.setObjective(obj, GRB.MAXIMIZE)

    for sid, norm_prob in group:
        s = inst.Scenarios[sid]
        for a in design_arcs:
            for k in commodities:
                lhs_design = lambda_vars[sid, k.Index, a.Origin.Index] - lambda_vars[sid, k.Index, a.Destination.Index] \
                      - mu_vars[sid, a.Index] - a.VariableCost * gamma_vars[sid]
                model.addConstr(lhs_design <= norm_prob * (1 - beta) * a.VariableCost)

        for a in dummy_arcs:
            for k in commodities:
                lhs_dummy = lambda_vars[sid, k.Index, a.Origin.Index] - lambda_vars[sid, k.Index, a.Destination.Index] \
                      - costdummy * gamma_vars[sid]
                model.addConstr(lhs_dummy <= norm_prob * (1 - beta) * costdummy)

        model.addConstr((1 - alpha) * gamma_vars[sid] <= beta * norm_prob)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError("Group dual subproblem not solved optimally.")

    lambda_vals = {(sid, k, n): lambda_vars[sid, k, n].X for (sid, k, n) in lambda_vars.keys()}
    mu_vals = {(sid, a): mu_vars[sid, a].X for (sid, a) in mu_vars.keys()}
    gamma_vals = {sid: gamma_vars[sid].X for sid in gamma_vars.keys()}

    return model.ObjVal, {"lambda": lambda_vals, "mu": mu_vals, "gamma": gamma_vals}


def solveBenders_grouped(inst, grouped_scenarios, intralevel_weights, alpha, beta, costdummy, time_limit):
    design_arcs = list(inst.Arcs.values())
    num_groups = len(grouped_scenarios)

    LB, UB = -float("inf"), float("inf")
    tolerance = 1e-6
    tolerance_opt_cut = 1e-5
    start_time = time.time()

    master, x, tau, theta = master_problem_grouped(inst, num_groups, intralevel_weights, alpha, beta)
    it, num_opt_cut = 1, 0
    LB_list, UB_list, abs_gap_list, rel_gap_list = [], [], [], []

    while time.time() - start_time < time_limit:
        print(f"\n--- Benders Iteration {it} ---")
        master.optimize()
        if master.status != GRB.OPTIMAL:
            raise RuntimeError("Master not optimal.")

        x_vals = {a.Index: x[a.Index].X for a in design_arcs}
        tau_val = tau.X
        LB = max(LB, master.ObjVal)
        current_UB = sum(a.FixedCost * x_vals[a.Index] for a in design_arcs) + beta * tau_val

        total_second_stage = 0.0

        for g, group in enumerate(grouped_scenarios):
            q_val, duals = dual_subproblem_grouped(inst, group, x_vals, tau_val, alpha, beta, costdummy)
            total_second_stage += intralevel_weights[g] * q_val

            lam = duals["lambda"]
            mu = duals["mu"]
            gamma = duals["gamma"]

            cut_expr = 0.0
            for sid, _ in group:
                s = inst.Scenarios[sid]
                part = quicksum(
                        s.CommodityDemand[k.Index] * (
                            lam[sid, k.Index, k.Origin.Index] - lam[sid, k.Index, k.Destination.Index]
                        ) for k in inst.Commodities.values()
                    ) - quicksum(
                        s.ArcCapacity[a.Index] * x[a.Index] * mu[sid, a.Index] for a in design_arcs
                    ) - tau * gamma[sid]
                
                cut_expr += part

            if theta[g].X < q_val - tolerance_opt_cut:
                master.addConstr(cut_expr <= theta[g], name=f"group_cut_{g}_it{it}")
                num_opt_cut += 1

        current_UB += total_second_stage
        UB = min(UB, current_UB)

        abs_gap = abs(UB - LB)
        rel_gap = abs(UB - LB) / (abs(UB) + 1e-6)
        print(f"LB: {LB:.4f}, UB: {UB:.4f}, AbsGap: {abs_gap:.8f}, RelGap: {rel_gap:.8f}")

        LB_list.append(LB)
        UB_list.append(UB)
        abs_gap_list.append(abs_gap)
        rel_gap_list.append(rel_gap)

        if abs_gap < tolerance or rel_gap < tolerance:
            break
        it += 1

    x_sol = {a.Index: int(round(x[a.Index].X)) for a in design_arcs}
    return master, x_sol, tau, LB_list, UB_list, abs_gap_list, rel_gap_list, num_opt_cut


def solveBenders_grouped_algo(inst, grouped_scenarios_lower_level, intralevel_weights_lower_level, grouped_scenarios_upper_level, intralevel_weights_upper_level, interlevel_weights, alpha, beta, costdummy, time_limit, scenario_group_map_upper_level):
    design_arcs = list(inst.Arcs.values())
    num_groups_lower_level = len(grouped_scenarios_lower_level)
    num_groups_upper_level = len(grouped_scenarios_upper_level)

    LB, UB = -float("inf"), float("inf")
    tolerance = 1e-6
    tolerance_opt_cut = 1e-5
    start_time = time.time()

    master, x, tau, theta = master_problem_grouped(inst, num_groups_upper_level, intralevel_weights_upper_level, alpha, beta)
    
    it, num_opt_cut = 1, 0
    LB_list, UB_list, abs_gap_list, rel_gap_list = [], [], [], []

    while time.time() - start_time < time_limit:
        print(f"\n--- Benders Iteration {it} ---")
        master.optimize()
        if master.status != GRB.OPTIMAL:
            raise RuntimeError("Master not optimal.")

        x_vals = {a.Index: x[a.Index].X for a in design_arcs}
        tau_val = tau.X
        LB = max(LB, master.ObjVal)
        current_UB = sum(a.FixedCost * x_vals[a.Index] for a in design_arcs) + beta * tau_val

        total_second_stage = 0.0

        cut_expr_lower_level = [0.0 for _ in range(num_groups_lower_level)]
        
        for g, group in enumerate(grouped_scenarios_lower_level):
            q_val, duals = dual_subproblem_grouped(inst, group, x_vals, tau_val, alpha, beta, costdummy)
            total_second_stage += intralevel_weights_lower_level[g] * q_val

            lam = duals["lambda"]
            mu = duals["mu"]
            gamma = duals["gamma"]

            cut_expr_lower_level[g] = 0.0
            part = 0.0
            for sid, _ in group:
                s = inst.Scenarios[sid]
                part = quicksum(
                        s.CommodityDemand[k.Index] * (
                            lam[sid, k.Index, k.Origin.Index] - lam[sid, k.Index, k.Destination.Index]
                        ) for k in inst.Commodities.values()
                    ) - quicksum(
                        s.ArcCapacity[a.Index] * x[a.Index] * mu[sid, a.Index] for a in design_arcs
                    ) - tau * gamma[sid]
                
                cut_expr_lower_level[g] += part


        aggregated_coeffs = {g: defaultdict(float) for g in range(num_groups_upper_level)}
        aggregated_constants = {g: 0.0 for g in range(num_groups_upper_level)}


        for g_lower, group in enumerate(grouped_scenarios_lower_level):
            first_sid = group[0][0]
            upper_gid = scenario_group_map_upper_level[first_sid]
            weight = interlevel_weights[g_lower]

            expr = cut_expr_lower_level[g_lower]

            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                aggregated_coeffs[upper_gid][var] += weight * coeff 
            aggregated_constants[upper_gid] += weight * expr.getConstant()

        aggregated_cuts = {}
        for g_upper in range(num_groups_upper_level):
            expr = LinExpr()
            for var, coeff in aggregated_coeffs[g_upper].items():
                expr.addTerms(coeff, var)
            expr.addConstant(aggregated_constants[g_upper]) 
            aggregated_cuts[g_upper] = expr

        for g_upper in range(num_groups_upper_level):
            master.addConstr(aggregated_cuts[g_upper] <= theta[g_upper],
                             name=f"benders_cut_uppergroup_{g_upper}_it{it}")
            num_opt_cut += 1

        print(f"Added {num_groups_upper_level} aggregated cuts to upper-level master.")

        current_UB += total_second_stage
        UB = min(UB, current_UB)

        abs_gap = abs(UB - LB)
        rel_gap = abs(UB - LB) / (abs(UB) + 1e-6)
        print(f"LB: {LB:.4f}, UB: {UB:.4f}, AbsGap: {abs_gap:.8f}, RelGap: {rel_gap:.8f}")

        LB_list.append(LB)
        UB_list.append(UB)
        abs_gap_list.append(abs_gap)
        rel_gap_list.append(rel_gap)

        if abs_gap < tolerance or rel_gap < tolerance:
            break
        it += 1
    

    x_sol = {a.Index: int(round(x[a.Index].X)) for a in design_arcs}

    return master, x_sol, tau, LB_list, UB_list, abs_gap_list, rel_gap_list, num_opt_cut



