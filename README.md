# PPSO-for-CCMPO
Particle Swarm Optimization (PSO) applied to solve the Cardinality Constrained Markowitz Portfolio Optimization (CCMPO) problem.


This work retraces the logical process behind the solid PSO model proposed by [Deng, Lin, and Lo (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0957417411014527), and aims to replicate their study, while also critically examining its strengths and limitations.

--- 

1. Portfolio Optimization  
2. Cardinality-Constrained Markowitz Portfolio Optimization (CCMPO)
3. Performance Evaluation  
4. Particle Swarm Optimization (PSO) Models  
5. Results  
6. Limitations and Discussion  
7. Proposed PSO for CCMPO Problems  

---

## 1. Portfolio Optimization

Portfolio optimization aims to select the best combination of assets to either:

- **Maximize return** for a given level of risk, or
- **Minimize risk** for a given level of return.
-   
$\text{Minimize} \quad \lambda \left[ \sum_{i=1}^{N} \sum_{j=1}^{N} x_i x_j \sigma_{ij} \right] - (1 - \lambda) \left[ \sum_{i=1}^{N} x_i \mu_i \right]$

$\text{subject to} \sum_{i=1}^{N} x_i = 1$

$0 \leq x_i \leq 1, \quad i = 1, \ldots, N$


One classical way of doing this is with the Markowitz's mean-variance optimization framework.

---

## 2. Cardinality-Constrained Markowitz Portfolio Optimization (CCMPO)

The CCMPO adds a **cardinality constraint** to the classic Markowitz model:

- Exactly **K assets** must be included in the final portfolio.

This leads to a combinatorial optimization problem that is **non-convex and NP-hard**.

$\sum_{i=1}^{N} z_i = K$

$z_i= $
- $1 \quad \text{if asset } i \text{ is included in the portfolio}$
- $0 \quad \text{otherwise}$


$\epsilon z_i \leq x_i \leq \delta z_i, \quad i = 1, \ldots, N$

$z_i \in [0, 1], \quad i = 1, \ldots, N$

---

## 3. Proposed PSO Approach for CCMPO

### Objective:
> To mitigate stagnation during the initial search phase of the PSO algorithm.

The following enhancements were implemented:

---

### 3.1 Reflection Strategy  
**Reference:** [Paterlini & Krink (2006)](https://www.sciencedirect.com/science/article/pii/S0167947304003962?casa_token=wUdVC5asyv0AAAAA:Ca0fEgmuH6hQTQSozKqcC91po00mFKsTYjN6Ral3kR6ENaqzoK8fiD4kMMFy3krr_LQYSw7w2x38)  

To prevent new positions from leaving the search space and causing stagnation towards local optima, the reflection strategy reflects positions back into the search space if they exceed boundaries, improving solution quality by facilitating exploration of a wider search area and escape from local minima.

$x_{i,j}^t = x_{i,j}^t + 2(x_j^l - x_{i,j}^t) \quad \text{if} \quad x_{i,j}^t < x_j^l$

$x_{i,j}^t = x_{i,j}^t - 2(x_{i,j}^t - x_j^u) \quad \text{if} \quad x_{i,j}^t > x_j^u$

$x_{i,j}^t = x_j^l, \quad \text{if} \quad x_{i,j}^t < x_j^l$

$x_{i,j}^t = x_j^u \quad \text{if} \quad x_{i,j}^t > x_j^u$

---

### 3.2 Cardinality Constraint Handling

To handle cardinality constraints, where K is the desired number of assets in the portfolio, assets are added or removed from a set Q based on whether the number of assets after updating positions, K_new ≠ K. 

- For cases where K_new > K, the study removes he smallest assets and the remaining ones are updated,
- while for K_new ≤ K, it randomly adds assets from Q with minimum proportional values. 

$x_i = \epsilon_i + \frac{s_i}{\sum_{j \in Q, s_j > \epsilon_i} s_i} \left( 1 - \sum_{j \in Q} \epsilon_j \right)
$

---

### 3.3 Linearly Decreasing Inertia Weights  
Reference: [Tripathi, Bandyopadhyay & Pal (2007)](https://www.sciencedirect.com/science/article/pii/S0020025507003155?casa_token=woihjR8-WGoAAAAA:Whl2dqiHHjwGJumMYLsS5v_qVFjVzCKLDaBjXx12CuSUwvvB-0ezWoCFEFwZJ67cMKAIySOVIrZq)

The inertia weight w in PSO controls the balance between exploration and exploitation, with high values promoting global exploration and low values emphasizing local exploitation. 
In complex problems like CCMPO, where both strategies are crucial, a time-varying w. This approach linearly reduces w over time, allowing particles to explore extensively initially and exploit gradually as the search progresses. Typically, w starts at 0.9 and ends at 0.4, adapting to the algorithm's evolving understanding of the search space throughout its execution.


$w(t) = (w(0) - w(n_t)) \cdot \frac{(n_t - t)}{n_t} + w(n_t)
$

Where:

- $w_{\text{start}} = 0.9$
- $w_{\text{end}} = 0.4$
- $T$ = total number of iterations
- $t$ = current iteration

---

### 3.4 Dynamic Acceleration Coefficients  
Reference: [Ratnaweera, Halgamuge & Watson (2004)](https://ieeexplore.ieee.org/abstract/document/1304846)

If c1 > c2, each particle has a stronger attraction to its own best position, and excessive wandering occurs. 
On the other hand, if c2 > c1, particles are most attracted to the global best position, which causes them to rush towards the optima prematurely.  
The coefficients vary over time:

$c_1(t) = (c_{1,min} - c_{1,max}) \frac{t}{n_t} + c_{1,max}$

$c_2(t) = (c_{2,max} - c_{2,min}) \frac{t}{n_t} + c_{2,min}$


---

### 3.5 Mutation Operator for Diversity  
Reference: [Tripathi et al., 2007](https://www.sciencedirect.com/science/article/pii/S0020025507003155?casa_token=ez5EKfyiVCYAAAAA:DoJMbff1ejB5GKbwXBZMt7Wi8eAbUs48w6n8-ScIOcARRd36j7wIMKTY97EnfSCOK8uYRVWgZ0y9)

To enhance diversity the algorithm randomly mutates a randomly selected variable g_k within defined limits, within specified upper and lower limits. The mutation process is governed by a random event flip resulting in either 0 or 1. 

$\Delta(t, x) = x \cdot \left( 1 - r^{\left( 1 - \frac{t}{max\_t} \right)^b} \right)$



---

### 3.6 Termination Criteria

Termination of the algorithm when no improvement over n repeated iterations.

---

## 4. Performance Evaluation

### Comparisons Performed:

- With classic PSO variants:  
  - Basic PSO  
  - PSO-DIV (diversity-enhanced)  
  - PSO-C (constriction coefficient)
  
- With other heuristics:  
  - Genetic Algorithms (GA)  
  - Simulated Annealing (SA)  
  - Tabu Search (TS)

### Evaluation Metrics:

- **Accuracy**:  
  - The standard deviation (risk) and return of the best solution for each k were used to compare standard efficient frontiers and measure percentage error respectively. 
  
- **Robustness**:  
  - Evaluated as the variance of a performance criterion over a number of trials. A smaller variance indicates that the algorithm's performance remains relatively consistent across different trials, suggesting greater stability and reliability.
  
- **Diversity**:  
  - Measured as the Euclidean distance between each particle's position and the average position across all particles in each dimension, then averages these distances across all particles.

---

## 5. Results

> Results indicated that the proposed PSO approach achieved higher accuracy, better robustness, and greater diversity than baseline PSO variants and other metaheuristics.

---

## 6. Limitations & Discussion

- Choice of parameters is predetermined in agreement with previous studies
  -  Consequences: no adaptation to the characteristics of the dataset. In particular with reference to swarm size.

- No possible corrections during training(number of iterations before forced termination and other parameters).

A more comprehensive evaluation may be considered
  - **Computational efficiency**
  - **Convergence speed**
  - **Scalability** to larger and more constrained portfolios

---

## 7. MATLAB Implementation

Implemented in MATLAB based on: Deng, Lin & Lo (2012)

Includes:

- Linearly decreasing inertia 
- Dynamic acceleration coefficients
- Cardinality constraints
- Mutation-based diversity enhancement
- Stagnation-based termination

---

