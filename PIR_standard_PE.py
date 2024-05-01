import pulp
import numpy as np

# Solver for winner determination problem Eq. (1) in https://core.ac.uk/download/pdf/132263314.pdf

def make_var(i):
    return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")

S = 100 # number of agents
b = np.random.random(size = (S)) # bid matrix (one bid per agent for the moment)
q = np.ones(S)
for i in range(len(b)):
    b[i] = b[i] - 0.5
    if b[i] < 0:
        q[i] = 1
    else:
        q[i] = -1

prob = pulp.LpProblem("Winner", pulp.LpMaximize)

wvars = [make_var(i) for i in range(S)]

prob += pulp.lpSum(b[i]*wvars[i] for i in range(S))
prob += pulp.lpSum(q[i]*wvars[i] for i in range(S)) >= 0 # note that since only one product, there is only one constraint here
# note that since one bid per agent, no constraint to enforce maximum one bid accepted
for i in range(S):
    prob += wvars[i] <= 1

pulp.LpStatus[prob.solve()]
#print(pulp.value(prob.objective), [pulp.value(wvars[i]) for i in range(S)])
#print(b)