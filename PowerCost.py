import pulp

# Plants, Products, Firms
plants = ["Jesolo", "Cosenza", "Cagliari"]
products = ["deckchair", "beach umbrella", "sun bed", "director's chair", "frisbee"]
firms = ["Fiumicino", "Genova", "Rimini", "Bari", "Cagliari", "Palermo", "Milano", "Salerno"]

# Demand data for each (product, firm) combination
demand = {
    ("deckchair", "Fiumicino"): 1128.2, ("deckchair", "Genova"): 1257.75, 
    ("deckchair", "Rimini"): 1215.82, ("deckchair", "Bari"): 1187.62, 
    ("deckchair", "Cagliari"): 1454.82, ("deckchair", "Palermo"): 1475.92, 
    ("deckchair", "Milano"): 1147.99, ("deckchair", "Salerno"): 1023.14,
    
    ("beach umbrella", "Fiumicino"): 1482.93, ("beach umbrella", "Genova"): 1011.23, 
    ("beach umbrella", "Rimini"): 1147.99, ("beach umbrella", "Bari"): 1023.14, 
    ("beach umbrella", "Cagliari"): 1248.89, ("beach umbrella", "Palermo"): 1008.76, 
    ("beach umbrella", "Milano"): 1479.57, ("beach umbrella", "Salerno"): 1027,
    
    ("sun bed", "Fiumicino"): 1122.46, ("sun bed", "Genova"): 1489.42, 
    ("sun bed", "Rimini"): 1479.57, ("sun bed", "Bari"): 1027, 
    ("sun bed", "Cagliari"): 1107.56, ("sun bed", "Palermo"): 1318.78, 
    ("sun bed", "Milano"): 1286.38, ("sun bed", "Salerno"): 1354.04,
    
    ("director's chair", "Fiumicino"): 1498.08, ("director's chair", "Genova"): 1318.78, 
    ("director's chair", "Rimini"): 1286.38, ("director's chair", "Bari"): 1354.04, 
    ("director's chair", "Cagliari"): 1000.04, ("director's chair", "Palermo"): 1088.28, 
    ("director's chair", "Milano"): 1187.62, ("director's chair", "Salerno"): 1454.82,
    
    ("frisbee", "Fiumicino"): 1111.23, ("frisbee", "Genova"): 1088.28, 
    ("frisbee", "Rimini"): 1187.62, ("frisbee", "Bari"): 1454.82, 
    ("frisbee", "Cagliari"): 1475.92, ("frisbee", "Palermo"): 1147.99, 
    ("frisbee", "Milano"): 1023.14, ("frisbee", "Salerno"): 1248.89
}

# Parameters for power consumption and limits
unit_consumption = {
    ("Jesolo", "deckchair"): 0.5, ("Jesolo", "beach umbrella"): 0.7, ("Jesolo", "sun bed"): 0.6,
    ("Jesolo", "director's chair"): 0.5, ("Jesolo", "frisbee"): 0.1,
    ("Cosenza", "deckchair"): 0.7, ("Cosenza", "beach umbrella"): 0.9, ("Cosenza", "sun bed"): 0.8,
    ("Cosenza", "director's chair"): 0.7, ("Cosenza", "frisbee"): 0.1,
    ("Cagliari", "deckchair"): 0.7, ("Cagliari", "beach umbrella"): 1.0, ("Cagliari", "sun bed"): 0.9,
    ("Cagliari", "director's chair"): 0.7, ("Cagliari", "frisbee"): 0.1
}

W_max = {"Jesolo": 56286.7, "Cosenza": 6940.54, "Cagliari": 8100.66}
T_low = 0.1
T_high = 0.2

# Initialize the model
model = pulp.LpProblem("Minimize_Power_Cost", pulp.LpMinimize)

# Decision Variables
x = pulp.LpVariable.dicts("Production", [(i, j) for i in plants for j in products], lowBound=0, cat='Continuous')
p = pulp.LpVariable.dicts("PowerConsumption", [(i, j) for i in plants for j in products], lowBound=0, cat='Continuous')
E = pulp.LpVariable.dicts("ExcessPower", plants, lowBound=0, cat='Continuous')  # Auxiliary variable for excess power

# Objective Function: Minimize power cost
model += pulp.lpSum(
    T_low * W_max[i] + T_high * E[i]  # Low rate up to W_max and high rate for excess
    for i in plants
), "Total_Power_Cost"

# Constraints
# 1. Power consumption for each product matches production * unit consumption
for i in plants:
    for j in products:
        model += p[(i, j)] == unit_consumption[(i, j)] * x[(i, j)], f"Power_Consumption_{i}_{j}"

# 2. Total power consumption per plant
for i in plants:
    model += pulp.lpSum(p[(i, j)] for j in products) <= W_max[i] + E[i], f"TotalPowerLimit_{i}"

# 3. Excess power constraint: E[i] should only be positive when total power > W_max[i]
for i in plants:
    model += E[i] >= pulp.lpSum(p[(i, j)] for j in products) - W_max[i], f"ExcessPower_{i}"

# 4. Demand fulfillment constraint for each product-firm pair
# Ensure that the total production across all plants meets the demand for each product-firm combination
for j in products:
    for k in firms:
        model += pulp.lpSum(x[(i, j)] for i in plants) >= demand[(j, k)], f"Demand_{j}_{k}"

# Solve the model
model.solve()

# Output the results
print("Status:", pulp.LpStatus[model.status])
print("Total Power Cost:", pulp.value(model.objective))
for i in plants:
    for j in products:
        for k in firms:
            print(f"Production of {j} at {i} for {k}: {x[(i, j)].varValue}")

total_decision_variables = len(model.variables())

print("Total number of decision variables:", total_decision_variables)
