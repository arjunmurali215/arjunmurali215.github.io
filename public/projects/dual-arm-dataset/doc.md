---
title: "Dual-Arm Grasp Dataset Generator"
date: "2025-08-21"
excerpt: "Generating a dual-arm manipulation dataset for small objects in cluttered scenes."
coverImage: "./assets/dual1.mp4"
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

---

## Phase 1: Cleaning the Data (NMS)

GraspNet-1B is fantastic, but it provides *too many* grasps. A single good grasp point might have hundreds of slightly varied orientations annotated around it. If we try to pair every single one of these, the computation time grows quadratically ($O(N^2)$).

To solve this, I used **Non-Maximum Suppression (NMS)**.
We calculate a distance metric between grasps that combines both their translation (position) and rotation (orientation):

$$d(g_i, g_j) = \|\mathbf{t}_i - \mathbf{t}_j\|_2 + \alpha \cdot d_{rot}(\mathbf{R}_i, \mathbf{R}_j)$$


If two grasps are physically too close and oriented similarly (within 3cm or 30 degrees), we keep the one with the higher quality score and discard the other. This drastically reduces the search space while preserving grasp diversity.

---

## Phase 2: The Pairing & The "Small Object" Problem

For an object with $n$ valid single-arm grasps, the number of possible dual-arm pairs is $\frac{n(n-1)}{2}$. For 500 grasps, that’s over 124,000 pairs to check.

This is where my implementation diverges from methods that focus on large objects. When grasping a small object (like a mug or a drill), the two grippers are dangerously close to each other. We need rigorous geometric pruning before we even think about physics.

I implemented two specific pruning criteria:

1.  **Wrist-to-Wrist Distance ($d_{w2w}$):** We project the gripper position backward along its approach vector to simulate the wrist/arm location. If the wrists are too close (< 0.1m), the arms will collide.
2.  **Center-to-Center Distance ($d_{c2c}$):** We ensure the grasp points aren't identical.
3.  **Axis Alignment:** We penalize pairs where the grippers have approach vectors too close to intersection. This makes a very significant difference for small objects.
---

## Phase 3: The Physics (Force Closure)

Once we have a geometrically valid pair, we have to answer the most important question: **Will it stick?**

This is determined by **Force Closure**—the ability of the grasp to resist arbitrary external wrenches (forces and torques).

### Step A: Finding Contact Points
We can't assume the gripper touches the object exactly at the center. I use **Signed Distance Functions (SDF)** to trace a "line of action" from the gripper jaws. The exact point where this line intersects the zero-level set of the object's SDF is our contact point.

### Step B: The Optimization
For a dual-arm grasp, we have 4 contact points (2 per gripper). We need to determine if there exists a set of contact forces that can balance out gravity and external disturbances without slipping.

This is an optimization problem. We minimize the error between the applied forces and the external wrench, subject to the friction cone constraints (Coulomb friction):

$$
\begin{aligned}
\min_{\mathbf{f}} \quad & \|\mathbf{G}\mathbf{f} + \mathbf{w}_{ext}\|_2 \\
\text{subject to} \quad & \|\mathbf{f}^{tan}\|_2 \leq \mu \cdot f^{normal}
\end{aligned}
$$

I used a friction coefficient $\mu$ of 0.3-0.4. If the optimization finds a solution where the residual is near zero ($< 10^{-5}$), the grasp is verified as stable.

---

## Results and Conclusion

The output of this pipeline is a set of verified dual-arm grasp configurations, scored by their ability to resist external forces.

By shifting the base data from DexNet to GraspNet-1B and implementing better collision detection, this generator allows us to create datasets for **cluttered scenes containing small objects**. This can be used toward training neural networks that can look at a messy table via a depth camera and decide how to use two hands to pick up an object safely.

### References
1. **GraspNet-1Billion**: Fang, H., et al. (CVPR 2020)
2. **DG16M**: A Large-Scale Dual-Arm Grasp Dataset