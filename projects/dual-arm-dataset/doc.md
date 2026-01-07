---
title: "Dual-Arm Grasp Dataset Generator"
date: "2025-08-21"
excerpt: "Generating a dual-arm manipulation dataset for small objects in cluttered scenes."
coverImage: "./assets/dual1.mp4"
repo: "https://github.com/arjunmurali215/dual-arm-dataset"
---

# Dual-Arm Grasp Dataset Generator


**Abstract:** While single-arm robotic grasping has seen massive datasets and benchmarks, dual-arm manipulation remains data-scarce. This post details my implementation of a pipeline that transforms single-arm grasp data into verified, force-closure dual-arm pairs. By adapting the logic from state-of-the-art research for cluttered, small-object scenarios, I aim to enable learning-based approaches for coordinated bimanual manipulation.

![Example generated dual-arm grasp pair](assets/dual1.mp4)

---

## The Motivation

I recently read the **DG16M** paper, which presented a massive dataset for dual-arm grasping. It was an inspiring piece of work, but as I dug into the methodology, I noticed two specific constraints that I wanted to overcome:

1.  **The "Large Object" Bias:** DG16M focused primarily on large objects (tables, boxes, bins). Because the objects were huge, collision avoidance between the two arms was trivial—simple distance pruning worked because the arms were naturally far apart. I wanted a system that could handle **smaller objects**, where the arms are forced into close proximity, making collision management critical.
2.  **Model vs. Scene:** DG16M generated grasps using **DexNet** as a base, which relies on clean 3D object models. I wanted to build my generation pipeline on top of **GraspNet-1B**. This dataset provides dense grasps in cluttered, real-world scenes captured with RGB-D cameras. By building on this, any network trained on my dataset can learn directly from depth images rather than requiring perfect CAD models.

---

## System Overview

The goal is straightforward but computationally heavy: Take a scene full of valid single-arm grasps and find every combination of two grasps that results in a stable, collision-free dual-arm hold.

The pipeline I built moves through four distinct phases:
1.  **Preprocessing:** Cleaning up the noisy GraspNet-1B data.
2.  **Pairing:** The combinatorial explosion of matching left and right hands.
3.  **Pruning:** Geometric checks to ensure the robots don't smash into each other.
4.  **Verification:** The physics math (Force Closure) to ensure the object doesn't fall.

### Grasp Representation

Before diving into the pipeline, it's important to understand how grasps are represented. Each parallel-jaw grasp is encoded as a 17-dimensional vector:

$$
\mathbf{g} = [s, w, h, d, \mathbf{R}, \mathbf{t}, o]^T \in \mathbb{R}^{17}
$$

where:
- $s \in [0, 1]$: Grasp quality score from GraspNet
- $w$: Gripper opening width (meters)
- $h$: Gripper finger height
- $d$: Approach depth
- $\mathbf{R} \in SO(3)$: Rotation matrix (9 elements, flattened) defining the gripper orientation
- $\mathbf{t} \in \mathbb{R}^3$: Translation vector (grasp center position)
- $o$: Object ID

The rotation matrix $\mathbf{R} = [\mathbf{x} \; | \; \mathbf{y} \; | \; \mathbf{z}]$ defines the grasp frame where **x** is the approach direction, **y** is the grasp axis (from jaw to jaw), and **z** completes the right-handed frame. The gripper jaw endpoints are computed as:

$$
\mathbf{g}_1 = \mathbf{t} - \frac{w}{2} \cdot \mathbf{y}, \quad \mathbf{g}_2 = \mathbf{t} + \frac{w}{2} \cdot \mathbf{y}
$$

---

## Phase 1: Cleaning the Data (NMS)

GraspNet-1B is fantastic, but it provides *too many* grasps. A single good grasp point might have hundreds of slightly varied orientations annotated around it. If we try to pair every single one of these, the computation time grows quadratically ($O(N^2)$).

To solve this, I used **Non-Maximum Suppression (NMS)**.
We calculate a distance metric between grasps that combines both their translation (position) and rotation (orientation):

$$
d(g_i, g_j) = \|\mathbf{t}_i - \mathbf{t}_j\|_2 + \alpha \cdot d_{rot}(\mathbf{R}_i, \mathbf{R}_j)
$$

The rotational distance is computed using the geodesic distance on the rotation manifold:

$$
d_{rot}(\mathbf{R}_i, \mathbf{R}_j) = \arccos\left(\frac{\text{tr}(\mathbf{R}_i^T \mathbf{R}_j) - 1}{2}\right)
$$

This gives us the angle of rotation needed to align the two grippers. With $\alpha = 0.00025$ as a weight factor, we balance translational and rotational components. If two grasps are physically too close (within 3cm) and oriented similarly (within 30° or $\pi/6$ radians), we keep the one with the higher quality score and discard the other. This drastically reduces the search space while preserving grasp diversity.

---

## Phase 2: The Pairing & The "Small Object" Problem

For an object with $n$ valid single-arm grasps, the number of possible dual-arm pairs is $\frac{n(n-1)}{2}$. For 500 grasps, that’s over 124,000 pairs to check.

This is where my implementation diverges from methods that focus on large objects. When grasping a small object (like a mug or a drill), the two grippers are dangerously close to each other. We need rigorous geometric pruning before we even think about physics.

I implemented three specific pruning criteria:

**1. Center-to-Center Distance ($d_{c2c}$):** This ensures grasps are sufficiently separated. The metric incorporates an axis alignment penalty:

$$
d_{c2c}(g_1, g_2) = \|\mathbf{t}_1 - \mathbf{t}_2\|_2 + \alpha \cdot d_{axis}
$$

where the axis distance penalizes parallel grasps:

$$
d_{axis} = \frac{2}{\pi} \arccos(|\mathbf{y}_1 \cdot \mathbf{y}_2|)
$$

When $|\mathbf{y}_1 \cdot \mathbf{y}_2| = 1$ (parallel grasps), $d_{axis} = 0$ and provides no penalty reduction. When $|\mathbf{y}_1 \cdot \mathbf{y}_2| = 0$ (perpendicular grasps), $d_{axis} = 1$ and provides maximum penalty. This is crucial because parallel grasps cannot resist torques about their common axis—a fundamental limitation in achieving force closure.

**2. Wrist-to-Wrist Distance ($d_{w2w}$):** We project the gripper position backward along its approach vector to simulate the wrist/arm location:

$$
\mathbf{w} = \mathbf{t} - d_{approach} \cdot \mathbf{x}
$$

where $d_{approach} = 0.1$ m and $\mathbf{x}$ is the approach direction (first column of $\mathbf{R}$). Then:

$$
d_{w2w}(g_1, g_2) = \|\mathbf{w}_1 - \mathbf{w}_2\|_2
$$

If the wrists are too close (< 0.1m), the robot arms will physically collide.

**3. Threshold Selection:** A pair is retained only if:

$$
\text{valid}(g_1, g_2) = (d_{c2c} > \tau_{c2c}) \land (d_{w2w} > \tau_{w2w})
$$

with primary thresholds $\tau_{c2c} = 0.06$ m and $\tau_{w2w} = 0.1$ m. If no pairs pass, we fall back to $\tau_{c2c} = 0.045$ m to ensure we get some candidates.

![Distances](assets/distances.jpeg)
---

## Phase 3: The Physics (Force Closure)

Once we have a geometrically valid pair, we have to answer the most important question: **Will it stick?**

This is determined by **Force Closure**—the ability of the grasp to resist arbitrary external wrenches (forces and torques).

### Step A: Finding Contact Points
We can't assume the gripper touches the object exactly at the center. I use **Signed Distance Functions (SDF)** to trace a "line of action" from the gripper jaws. 

For each gripper jaw, starting from the endpoint $\mathbf{g}_i$, the line extends along the grasp axis:

$$
\mathcal{L}_1 = \{\mathbf{g}_1 + \lambda \cdot \mathbf{y} \; | \; \lambda \in [0, w]\}
$$
$$
\mathcal{L}_2 = \{\mathbf{g}_2 - \lambda \cdot \mathbf{y} \; | \; \lambda \in [0, w]\}
$$

The SDF $\phi: \mathbb{R}^3 \rightarrow \mathbb{R}$ encodes the distance to the object surface, where $\phi(\mathbf{p}) < 0$ means the point is inside the object. Contact occurs at the first zero-crossing along the line:

$$
\mathbf{c} = \arg\min_{\mathbf{p} \in \mathcal{L}} \|\mathbf{p} - \mathbf{g}\| \quad \text{s.t.} \quad \phi(\mathbf{p}) \leq 0
$$

For a dual-arm grasp, we compute four contact points: $\mathbf{c}_1, \mathbf{c}_2$ from gripper 1 and $\mathbf{c}_3, \mathbf{c}_4$ from gripper 2.

### Step A.5: Surface Normals and Grasp Maps

Once we have contact points, we need surface normals. The normal at a contact point is the normalized gradient of the SDF:

$$
\mathbf{n} = \frac{\nabla \phi(\mathbf{c})}{\|\nabla \phi(\mathbf{c})\|}
$$

**Critical detail:** The normals must point **inward** (toward the object interior) for the contact model to work correctly.

Now comes the grasp map construction. For each contact $i$, I construct a contact frame $\mathbf{T}_i$ with rotation $\mathbf{R}_i = [\mathbf{t}_1 \; | \; \mathbf{t}_2 \; | \; \mathbf{n}_i]$ where $\mathbf{t}_1, \mathbf{t}_2$ are tangent vectors spanning the contact plane. The grasp map relates contact forces to object wrenches via the adjoint transformation:

$$
\mathbf{G}_i = \text{Ad}_{\mathbf{T}_i} \cdot \mathbf{B} \in \mathbb{R}^{6 \times 3}
$$

where the adjoint is:

$$
\text{Ad}_{\mathbf{T}} = \begin{bmatrix} \mathbf{R} & \mathbf{0} \\ [\mathbf{p}]_\times \mathbf{R} & \mathbf{R} \end{bmatrix} \in \mathbb{R}^{6 \times 6}
$$

and $[\mathbf{p}]_\times$ is the skew-symmetric cross-product matrix. For point contact with friction, the basis matrix is:

$$
\mathbf{B} = \begin{bmatrix} \mathbf{I}_{3\times3} \\ \mathbf{0}_{3\times3} \end{bmatrix}
$$

The combined grasp map for all four contacts is:

$$
\mathbf{G} = [\mathbf{G}_1 \; | \; \mathbf{G}_2 \; | \; \mathbf{G}_3 \; | \; \mathbf{G}_4] \in \mathbb{R}^{6 \times 12}
$$

A necessary condition for force closure is $\text{rank}(\mathbf{G}) = 6$. If this fails, we immediately reject the grasp pair.

### Step B: The Optimization
For a dual-arm grasp, we have 4 contact points (2 per gripper). We need to determine if there exists a set of contact forces that can balance out gravity and external disturbances without slipping.

This is formulated as a **Second-Order Cone Program (SOCP)**:

$$
\begin{aligned}
\min_{\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4} \quad & \|\mathbf{G}_1\mathbf{f}_1 + \mathbf{G}_2\mathbf{f}_2 + \mathbf{G}_3\mathbf{f}_3 + \mathbf{G}_4\mathbf{f}_4 + \mathbf{w}_{ext}\|_2 \\
\text{subject to} \quad & \|[f_{i,x}, f_{i,y}]^T\|_2 \leq \mu \cdot f_{i,z}, \quad i = 1,2,3,4 \\
& \|\mathbf{f}_i\|_2 \leq f_{max}, \quad i = 1,2,3,4
\end{aligned}
$$

The friction cone constraints encode **Coulomb friction**: tangential forces (in the contact plane) must not exceed $\mu$ times the normal force. These are second-order cone constraints—hence SOCP rather than standard linear programming.

The external wrench represents gravitational loading:

$$
\mathbf{w}_{ext} = \begin{bmatrix} 0 \\ 0 \\ -mg \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$

where I use $m = 10 \times$ actual object mass to test robustness. I used a friction coefficient $\mu$ of 0.3-0.4 (typical for rubber/plastic) and $f_{max} = 60$ N. 

A grasp achieves force closure if:

$$
\text{loss}^* = \|\mathbf{G}\mathbf{f}^* + \mathbf{w}_{ext}\|_2 < \epsilon
$$

where $\epsilon = 10^{-5}$ is my convergence threshold. The optimized forces $\mathbf{f}^*$ represent the contact forces in the contact frame, which I transform back to the object frame for visualization.

---

## Results and Conclusion

The output of this pipeline is a set of verified dual-arm grasp configurations, scored by their ability to resist external forces.

By shifting the base data from DexNet to GraspNet-1B and implementing better collision detection, this generator allows us to create datasets for **cluttered scenes containing small objects**. This can be used toward training neural networks that can look at a messy table via a depth camera and decide how to use two hands to pick up an object safely.

### References
1. **GraspNet-1Billion**: Fang, H., et al. (CVPR 2020)
2. **DG16M**: A Large-Scale Dual-Arm Grasp Dataset

