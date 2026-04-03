import random
import copy

try:
    from pysat.solvers import Solver
    _HAS_PYSAT = True
except Exception:
    Solver = None
    _HAS_PYSAT = False

class RandomizedDPLL:
    """
    A simple Boolean Satisfiability (SAT) solver based on the DPLL algorithm,
    modified to return a random SAT solution by randomizing variable branching
    order and polarity, as described in the SAFARI paper.
    """
    def __init__(self, cnf, num_vars):
        self.cnf = cnf
        self.num_vars = num_vars
        
    def _bcp(self, formula, assignment):
        """Boolean Constraint Propagation (Unit Propagation)."""
        new_formula = []
        unit_clauses = [c for c in formula if len(c) == 1]
        
        while unit_clauses:
            unit = unit_clauses[0][0]
            assignment.append(unit)
            
            updated_formula = []
            for clause in formula:
                if unit in clause:
                    continue  # Clause satisfied
                if -unit in clause:
                    new_clause = [l for l in clause if l != -unit]
                    if not new_clause:
                        return False, [] # Conflict
                    updated_formula.append(new_clause)
                else:
                    updated_formula.append(clause)
            
            formula = updated_formula
            unit_clauses = [c for c in formula if len(c) == 1]
            
        return True, formula

    def _get_unassigned(self, assignment):
        assigned_vars = set(abs(a) for a in assignment)
        return [v for v in range(1, self.num_vars + 1) if v not in assigned_vars]

    def solve_iterative(self):
        """
        Iterative DPLL solver to avoid RecursionError and improve performance.
        Uses a stack to manage the search state: (formula, assignment)
        """
        # Stack elements: (formula, assignment_so_far)
        stack = [(self.cnf, [])]
        
        while stack:
            # Pop the current state (DFS)
            formula, assignment = stack.pop()
            
            # 1. BCP (Boolean Constraint Propagation)
            status, formula = self._bcp(formula, assignment)
            
            if not status:
                continue # Conflict, backtrack (implicitly by popping next from stack)
            
            if not formula:
                return assignment # Found a solution!
            
            # 2. Branching
            unassigned = self._get_unassigned(assignment)
            if not unassigned:
                return assignment # Should verify full assignment
            
            # Strict SAFARI Randomization: Random Limit Logic
            # Optimization: Try to prioritize unassigned vars involved in remaining clauses
            # But SAFARI paper insists on uniform random for properties.
            var = random.choice(unassigned)
            polarity = random.choice([1, -1])
            guess = var * polarity
            
            # Push branches to stack. 
            # Note: We push the second branch first so it's processed later.
            # Branch 2: Assume -guess
            stack.append((formula + [[-guess]], assignment.copy()))
            
            # Branch 1: Assume guess (processed next iteration)
            stack.append((formula + [[guess]], assignment.copy()))
            
        return None

    def get_random_solution(self):
        # Use iterative solver
        sol = self.solve_iterative()
        
        if sol is None:
            return None
        # Fill strictly remaining unassigned variables with random truths
        assigned = set(abs(l) for l in sol)
        for i in range(1, self.num_vars + 1):
            if i not in assigned:
                sol.append(i if random.random() > 0.5 else -i)
        return set(sol)


class RandomizedPySAT:
    """
    High-performance SAT backend using python-sat.
    Keeps the same interface as RandomizedDPLL.
    """
    def __init__(self, cnf, num_vars, solver_name='glucose3'):
        if not _HAS_PYSAT:
            raise RuntimeError("python-sat is not installed")
        self.num_vars = num_vars
        self.solver = Solver(name=solver_name, bootstrap_with=cnf)

    def get_random_solution(self):
        # Randomized phase hints to emulate RandomDiagnosis behavior.
        phase_hints = [v if random.random() > 0.5 else -v for v in range(1, self.num_vars + 1)]
        try:
            self.solver.set_phases(phase_hints)
        except Exception:
            # Some backends may ignore phase control.
            pass

        is_sat = self.solver.solve()
        if not is_sat:
            return None

        model = self.solver.get_model()
        if model is None:
            return None

        model_set = set(model)
        assigned = set(abs(l) for l in model_set)
        # Ensure a full assignment over known vars.
        for v in range(1, self.num_vars + 1):
            if v not in assigned:
                model_set.add(v if random.random() > 0.5 else -v)
        return model_set

    def close(self):
        try:
            self.solver.delete()
        except Exception:
            pass

class SafariMBD:
    def __init__(self, circuit_simulator, max_tries=5, max_climb=8, sat_backend='auto'):
        self.simulator = circuit_simulator
        self.max_tries = max_tries
        self.max_climb = max_climb
        self.sat_backend = sat_backend
        self.cnf = []
        self.var_map = {}
        self.rev_var_map = {}
        self.var_count = 0
        self._sat_solver = None
        
    def _new_var(self, name):
        if name not in self.var_map:
            self.var_count += 1
            self.var_map[name] = self.var_count
            self.rev_var_map[self.var_count] = name
        return self.var_map[name]

    def convert_to_cnf(self, inputs, expected_obs):
        """WffToCNF: Converts the System Description (SD) and Observations to CNF."""
        if self._sat_solver and hasattr(self._sat_solver, 'close'):
            self._sat_solver.close()
        self._sat_solver = None

        self.cnf = []
        self.var_map.clear()
        self.rev_var_map.clear()
        self.var_count = 0
        
        # 1. Force primary inputs
        for inp_node, val in inputs.items():
            var = self._new_var(f"val_{inp_node}")
            self.cnf.append([var] if val == 1 else [-var])
            
        # 2. Encode Logic Gates with Health Variables (SD)
        # Assuming only stuck-at components for simplicity
        for node in self.simulator.G.nodes():
            n_type = self.simulator.G.nodes[node].get('type', 'UNKNOWN')
            if n_type == 'INPUT': continue
            
            y = self._new_var(f"val_{node}")
            preds = list(self.simulator.G.predecessors(node))
            if not preds: continue
            
            # Health variables: h=0 (healthy), h_sa0 (stuck-at-0), h_sa1 (stuck-at-1)
            h_sa0 = self._new_var(f"sa0_{node}")
            h_sa1 = self._new_var(f"sa1_{node}")
            self.cnf.append([-h_sa0, -h_sa1]) # Cannot be both sa0 and sa1
            
            # If healthy -> output matches logic
            x = [self._new_var(f"val_{p}") for p in preds]
            
            if n_type == 'AND':
                for xi in x:
                    self.cnf.append([h_sa0, h_sa1, -y, xi])
                self.cnf.append([h_sa0, h_sa1, y] + [-xi for xi in x])
            elif n_type == 'OR':
                for xi in x:
                    self.cnf.append([h_sa0, h_sa1, y, -xi])
                self.cnf.append([h_sa0, h_sa1, -y] + [xi for xi in x])
            elif n_type == 'NAND':
                for xi in x:
                    self.cnf.append([h_sa0, h_sa1, y, xi])
                self.cnf.append([h_sa0, h_sa1, -y] + [-xi for xi in x])
            elif n_type == 'NOR':
                for xi in x:
                    self.cnf.append([h_sa0, h_sa1, -y, -xi])
                self.cnf.append([h_sa0, h_sa1, y] + [xi for xi in x])
            elif n_type == 'NOT' or n_type == 'BUFF':
                if n_type == 'NOT':
                    self.cnf.append([h_sa0, h_sa1, -y, -x[0]])
                    self.cnf.append([h_sa0, h_sa1, y, x[0]])
                else: # BUFF
                    self.cnf.append([h_sa0, h_sa1, -y, x[0]])
                    self.cnf.append([h_sa0, h_sa1, y, -x[0]])
            elif n_type == 'XOR':
                # Correct XOR for 2-input: y = x[0] XOR x[1]
                self.cnf.append([h_sa0, h_sa1, -y, x[0], x[1]])
                self.cnf.append([h_sa0, h_sa1, -y, -x[0], -x[1]])
                self.cnf.append([h_sa0, h_sa1, y, x[0], -x[1]])
                self.cnf.append([h_sa0, h_sa1, y, -x[0], x[1]])
                
            # If stuck-at-0 -> y = 0
            self.cnf.append([-h_sa0, -y])
            # If stuck-at-1 -> y = 1
            self.cnf.append([-h_sa1, y])

        # 3. Force outputs (alpha / observations)
        for out_node, expected_val in expected_obs.items():
            out_var = self._new_var(f"val_{out_node}")
            self.cnf.append([out_var] if expected_val == 1 else [-out_var])

    def random_diagnosis(self):
        """RandomDiagnosis using selected SAT backend."""
        if self._sat_solver is None:
            backend = self.sat_backend
            if backend == 'auto':
                backend = 'pysat' if _HAS_PYSAT else 'dpll'

            if backend == 'pysat':
                self._sat_solver = RandomizedPySAT(self.cnf, self.var_count)
            else:
                self._sat_solver = RandomizedDPLL(self.cnf, self.var_count)

        sol = self._sat_solver.get_random_solution()
        
        if sol is None:
            return set() # No solution found, should not happen if observations are valid under some fault
            
        # Extract diagnosis: terms where health vars are TRUE (faulty)
        omega = set()
        for s in sol:
            if s > 0:
                name = self.rev_var_map[s]
                if name.startswith("sa0_"):
                    omega.add((name[4:], 0))
                elif name.startswith("sa1_"):
                    omega.add((name[4:], 1))
        return omega

    def improve_diagnosis(self, omega):
        """Randomly 'un-flip' a fault to improve cardinality."""
        if not omega: return set()
        omega_prime = set(omega)
        node_to_remove = random.choice(list(omega_prime))
        omega_prime.remove(node_to_remove)
        return omega_prime

    def check_consistency(self, inputs, expected_obs, omega):
        """SD ^ alpha ^ omega != bottom"""
        sim_out = self.simulator.simulate(inputs, fault=omega)
        for obs_node, obs_val in expected_obs.items():
            if sim_out.get(obs_node) != obs_val:
                return False
        return True

    def run(self, inputs, expected_obs):
        self.convert_to_cnf(inputs, expected_obs)
        R = set() # Set of minimal diagnoses
        
        for n in range(self.max_tries):
            omega = self.random_diagnosis()
            
            m = 0
            while m < self.max_climb:
                # If omega is empty, we cannot improve cardinality further (it is size 0)
                if not omega:
                    break
                    
                omega_prime = self.improve_diagnosis(omega)
                if self.check_consistency(inputs, expected_obs, omega_prime):
                    omega = omega_prime
                    
                    m = 0
                else:
                    m += 1
                    
            # Subsumption check
            is_sub = False
            for prev_omega in R:
                if prev_omega.issubset(omega):
                    is_sub = True
                    break
            
            if not is_sub:
                # Remove subsumed by new omega
                R = {o for o in R if not omega.issubset(o)}
                R.add(frozenset(omega))

        if self._sat_solver and hasattr(self._sat_solver, 'close'):
            self._sat_solver.close()
        self._sat_solver = None
                
        return R

# Usage Example:
# simulator = CircuitSimulator(G)
# safari = SafariMBD(simulator, M, N)
# result = safari.run(inputs, expected_obs)
