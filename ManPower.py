import pulp

# Plants, Products, and Firms
plants = ["Jesolo", "Cosenza", "Cagliari"]
products = ["deckchair", "beach umbrella", "sun bed", "director's chair", "frisbee"]
firms = ["Fiumicino", "Genova", "Rimini", "Bari", "Cagliari", "Palermo", "Milano", "Salerno"]

# Parameters
hours_per_product = {
    ("Jesolo", "deckchair"): 0.05, ("Jesolo", "beach umbrella"): 0.1, ("Jesolo", "sun bed"): 0.1,
    ("Jesolo", "director's chair"): 0.1, ("Jesolo", "frisbee"): 0.02,
    ("Cosenza", "deckchair"): 0.1, ("Cosenza", "beach umbrella"): 0.15, ("Cosenza", "sun bed"): 0.15,
    ("Cosenza", "director's chair"): 0.1, ("Cosenza", "frisbee"): 0.02,
    ("Cagliari", "deckchair"): 0.1, ("Cagliari", "beach umbrella"): 0.15, ("Cagliari", "sun bed"): 0.1,
    ("Cagliari", "director's chair"): 0.1, ("Cagliari", "frisbee"): 0.02
}

working_hours_per_worker = {"Jesolo": 140, "Cosenza": 120, "Cagliari": 130}
monthly_salary = {"Jesolo": 1500, "Cosenza": 1200, "Cagliari": 1250}
worker_bounds = {"Jesolo": (15, 30), "Cosenza": (20, 30), "Cagliari": (15, 35)}

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

# Initialize the model
model = pulp.LpProblem("Minimize_Labor_Cost", pulp.LpMinimize)

# Decision Variables
x = pulp.LpVariable.dicts("Production", [(i, j, k) for i in plants for j in products for k in firms], lowBound=0, cat='Continuous')
w = pulp.LpVariable.dicts("Workers", plants, lowBound=0, cat='Integer')

# Objective Function: Minimize labor cost
model += pulp.lpSum(monthly_salary[i] * w[i] for i in plants), "Total_Labor_Cost"

# Constraints
# 1. Labor Hours Constraint
for i in plants:
    model += pulp.lpSum(hours_per_product[(i, j)] * x[(i, j, k)] for j in products for k in firms) <= working_hours_per_worker[i] * w[i], f"Hours_Availability_{i}"

# 2. Worker Bounds at each plant
for i in plants:
    lb, ub = worker_bounds[i]
    model += w[i] >= lb, f"Min_Workers_{i}"
    model += w[i] <= ub, f"Max_Workers_{i}"

# 3. Demand Fulfillment Constraint for each product-firm combination
for j in products:
    for k in firms:
        # Check if the demand for this product-firm combination exists
        if (j, k) in demand:
            model += pulp.lpSum(x[(i, j, k)] for i in plants) >= demand[(j, k)], f"Demand_{j}_{k}"
        else:
            print(f"Warning: Demand for ({j}, {k}) is missing!")

# Solve the model
model.solve()

# Output the results
print("Status:", pulp.LpStatus[model.status])
print("Total Cost:", pulp.value(model.objective))
count = 0
for i in plants:
    print(f"Workers at {i}: {w[i].varValue}")
    for j in products:
        for k in firms:
            print(f"Production of {j} at {i} for {k}: {x[(i, j, k)].varValue}")
            count+=1
print(count)

total_decision_variables = len(model.variables())

print("Total number of decision variables:", total_decision_variables)
        
