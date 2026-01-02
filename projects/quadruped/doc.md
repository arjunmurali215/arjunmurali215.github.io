---
title: "Go2 Quadruped Locomotion"
date: "2024-08-20"
excerpt: "Reinforcement learning pipeline for quadruped locomotion using Genesis and PPO."
coverImage: "./assets/dog.mp4"
---



# Go2 Quadruped Locomotion

**Abstract:** This report details the implementation of a locomotion control pipeline for the Unitree Go2 quadruped. By leveraging the Genesis physics engine for high-fidelity simulation and the RSL-RL library for efficient training, I developed a policy capable of robust walking, running, and jumping. The system uses Proximal Policy Optimization (PPO) to learn complex gaits in parallelized environments.

![Go2 Locomotion](assets/dog.mp4)

---

## 1. Introduction: Learning to Walk

Last year, I started delving into quadrupeds. I then stumbled upon ETH Zurich's Robotic Systems Lab (RSL)'s `rsl_rl` library. I wanted to see if I could use that framework to implement teleop on the **Unitree Go2**, just for my learning.

The goal was straightforward: Use the existing tools to train a neural network that pilots the 12-DOF robot from scratch, enabling it to track velocity commands and perform dynamic maneuvers like jumping.

### 1.1 System Architecture

The project leverages existing robust tools to build a specific application:

| Component | Role |
|-----------|------|
| **Genesis** | The physics simulation backend. Handles rigid body dynamics, collisions, and contacts. |
| **RSL-RL** | The RL framework. I used this library to handle the PPO algorithm and training loop. |
| **QuadrupedEnv** | The task implementation. A custom environment I wrote to interface Genesis with the RSL-RL framework. |

Training is parallelized. Instead of running one robot at a time, I spawn **4096 environments** on the GPU. This allows the agent to collect experience steps much faster.

---

## 2. The Simulation World

### 2.1 The Robot Model

The Unitree Go2 is a 12-DOF robot (3 actuators per leg). It is loaded into the Genesis simulation from its URDF description.



The joint configuration follows a standard quadruped layout:
* **Abduction/Adduction (Hip)**
* **Hip Flexion/Extension (Thigh)**
* **Knee Flexion/Extension (Calf)**

I initialize the robot in a standing pose with joint angles pre-determined by me.

Control is achieved via **PD position control** at the joint level. The policy outputs residual target angles, which are added to the default standing pose.

---

## 3. Observation & Action Space

### 3.1 Observations ($O_t \in \mathbb{R}^{48}$)
The policy receives a 48-dimensional vector containing proprioceptive data and task commands:

* **Body State:** Base angular velocity, projected gravity vector (to know which way is down).
* **Joint State:** Positions (error relative to default) and velocities.
* **History:** The previous action taken (crucial for smoothness).
* **Commands:** Target velocities ($v_x, v_y, \omega_z$), target height, and jump flags.

### 3.2 Actions ($A_t \in \mathbb{R}^{12}$)
The network outputs 12 values corresponding to target joint positions.
$$q_{target} = q_{default} + 0.25 \cdot \text{Action}$$

To bridge the "sim-to-real" gap, I implemented a **one-step action delay**. The action computed at time $t$ is not applied until $t+1$, mimicking the communication latency of real hardware.

---

## 4. Rewards

### 4.1 Tracking Rewards
The primary objective is to follow the user's command.
* **Linear Velocity:** $r_{lin} = \exp(-\|v_{cmd} - v_{actual}\|^2 / \sigma^2)$
* **Angular Velocity:** Similar exponential kernel for yaw rate.

### 4.2 Survival & Style Penalties
To prevent the robot from flailing or damaging itself, I added:
* **Smoothness:** Penalize jerky changes in action ($\|a_t - a_{t-1}\|^2$).
* **Nominal Pose:** Penalize deviating too far from the natural standing posture.
* **Vertical Stability:** Penalize vertical body velocity when *not* jumping (we want walking, not bouncing).

### 4.3 The Jump Logic
Jumping is handled via a Finite State Machine (FSM) inside the reward function. When the "jump" command is active:
1.  **Preparation:** Penalties relax to allow crouching.
2.  **Peak:** Massive rewards for matching target jump height ($500 \times$ weight) and vertical velocity.
3.  **Landing:** Rewards switch to stabilizing the base.

---

## 5. Training with PPO

I trained the policy using Proximal Policy Optimization (PPO) provided by the **RSL-RL** library. The actor and critic networks are simple Multi-Layer Perceptrons (MLP) with ELU activations:
`Input (48) -> 512 -> 256 -> 128 -> Output`

**Training Stats:**
* **Environments:** 4096
* **Iterations:** 1500 (taking approx. 30-45 mins on an RTX 4050)

The massive batch size creates a very stable gradient estimate, allowing for aggressive learning rates without destabilizing the policy.

---

## 6. Evaluation & Teleoperation

After training, I validated the policy using a custom teleoperation script. This maps keyboard inputs to the command vector, allowing me to drive the robot around the simulation in real-time.

| Key | Command | Effect |
|-----|---------|--------|
| **W / S** | $v_x$ | Move Forward / Backward |
| **A / D** | $v_y$ | Strafe Left / Right |
| **Q / E** | $\omega_z$ | Turn Left / Right |
| **J** | Jump | Trigger jump maneuver |

The policy proved robust to external disturbances, recovering quickly from pushes and maintaining balance even when transitioning rapidly between forward running and lateral strafing.

---

## 7. Conclusion

This project successfully demonstrated a  locomotion pipeline for the Go2. It was a good learning experience for me.