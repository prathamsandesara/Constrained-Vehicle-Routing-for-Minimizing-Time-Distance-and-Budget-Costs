# Constrained Vehicle Routing for Minimizing Time, Distance, and Budget Costs

## 🧪 Simulation Tools & Setup

### 🧰 Technologies Used

- Python  
- SUMO (Simulation of Urban Mobility)  
- NetworkX  
- Matplotlib  
- SumoLib  

### 📂 Key Files

- `csr.net.xml`: SUMO road network file  
- `test.rou.xml`: Generated routes  
- `trips.trips.xml`: Random traffic input  
- `config.sumocfg`: SUMO configuration  

### 🧵 Graph Generation

- Nodes are parsed from the SUMO network  
- Directed edges with weighted distances are added  
- Graph is visualised with optimised route highlighted  

## 📌 Algorithms

### 📍 Deterministic A* Algorithm

- Chooses optimal path based on fixed cost and heuristic  
- Not suitable for dynamic real-world changes  

### 🎲 Randomized A* Algorithm

Adds probabilistic elements to:

- Node expansion strategy  
- Cost function: modeled via distributions  
- Heuristic function: adds noise and weights  

```python
h(n) = w1*T(n) + w2*F(n) + w3*B(n) + ε
```

This enhances exploration, adaptability, and realism.

## 🔁 Randomization Techniques

| Random Component       | Distribution     |
|------------------------|------------------|
| Travel Time            | Normal           |
| Fuel Cost              | Exponential      |
| Breakdown Probability  | Bernoulli        |
| Heuristic Noise        | Gaussian         |

The heuristic uses stochastic weights to balance route decisions.

## 📊 Results

### 💻 Computer Science Evaluation

| Aspect             | Deterministic A* | Randomized A*      |
|--------------------|------------------|---------------------|
| Time Complexity    | O(b^d)           | O(N log N)          |
| Traffic Modeling   | None             | Real-time supported |
| Graph Size         | Small (10x10)    | Large networks      |
| Adaptability       | Static           | Dynamic             |
| Realism            | Low              | High                |

### 🌍 Domain Insights

- Graphs show how traffic-aware randomized A* avoids congestion  
- Histograms show cost variations before and after traffic  
- Comparisons validate that while randomized A* has variability, it often finds lower cost routes due to broader exploration  

## 📐 Mathematical Bounds

### 📌 Markov's Inequality

Used to bound worst-case delays:

```math
P(T ≥ α) ≤ E[T] / α
```

### 📌 Chebyshev’s Inequality

Used to measure time deviation from average:

```math
P(|T − μ| ≥ kσ) ≤ 1 / k²
```

### 📌 Cauchy-Schwarz Inequality

Used to study correlation between travel time and fuel cost:

```math
|E[T * F]| ≤ √(E[T²] · E[F²])
```

These bounds improve predictability and reliability of routing decisions under uncertainty.

## 📁 Suggested Folder Structure

```
.
├── code/
│   ├── astar_randomized.py
│   ├── csr.net.xml
│   ├── test.rou.xml
│   ├── trips.trips.xml
│   └── config.sumocfg
├── images/
│   ├── graph.png
│   ├── traffic_simulation.png
│   ├── cost_histogram_before.png
│   └── cost_histogram_after.png
├── report/
│   └── CSE400_its_3_ProjectReport.pdf
└── README.md
```



 

## 🚀 Motivation

Urban expansion and increasing traffic demand smarter, adaptive route planning. Conventional algorithms focus only on shortest path or travel time without considering dynamic and probabilistic conditions. Our project aims to:

- 📉 Minimize travel time, cost, and fuel usage  
- 📦 Improve delivery and logistics efficiency  
- 🔄 Adapt to changing real-time traffic conditions  

---


## 📈 Methodology

### 🔢 Random Variables Modeled

| Parameter             | Type              | Distribution         |
|----------------------|-------------------|----------------------|
| Travel Time (T)      | Continuous        | Normal Distribution  |
| Fuel Cost (F)        | Continuous        | Exponential          |
| Traffic Congestion   | Continuous        | Normal Distribution  |
| Distance (D)         | Continuous        | Normal Distribution  |
| Vehicle Breakdown (B)| Discrete          | Bernoulli            |
` if the route from node i to j is selected, else `0`.


## 🧩 Applications

This routing algorithm can benefit various real-world sectors:

1. **Delivery Services** – Flipkart, Amazon, Swiggy, Uber  
2. **Emergency Healthcare** – Timely delivery of samples or supplies  
3. **Fleet Management Systems** – Smart route suggestions using traffic/weather data  
4. **Public Transport** – Optimized scheduling and rerouting using GPS and traffic inputs  

---
## 💡 Future Enhancements

- Integration with live GPS APIs (e.g., Google Maps API)  
- Support for electric vehicles (charging station constraints)  
- Enhanced UI for real-time route visualizations  
- Deployment as a web-based application  

## 🎓 Course Information

**Course Title**: CSE400 – Fundamentals of Probability in Computing  
**School**: School of Engineering and Applied Science (SEAS), Ahmedabad University  
**Instructor**: Prof. Dhaval Patel 
**Semester**: Semester 4 - Winter 2025 
