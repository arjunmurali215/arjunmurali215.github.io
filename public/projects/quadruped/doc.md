---
title: "Go2 Quadruped Locomotion"
date: "2025-04-20"
excerpt: "Reinforcement learning pipeline for quadruped locomotion using Genesis and PPO."
coverImage: "./assets/dog.mp4"
repo: "https://github.com/arjunmurali215/ppo-doggy"
---

# Go2 Quadruped Locomotion

**Abstract:** This report details the implementation of a locomotion control pipeline for the Unitree Go2 quadruped. By leveraging the Genesis physics engine for high-fidelity simulation and the RSL-RL library for efficient training, I developed a policy capable of robust walking, running, and jumping. The system uses Proximal Policy Optimization (PPO) to learn complex gaits in parallelized environments.

![Go2 Locomotion](assets/dog.mp4)

---

## 1. Introduction: Learning to Walk

Sometime back, I started delving into quadrupeds. I then stumbled upon ETH Zurich's Robotic Systems Lab (RSL)'s `rsl_rl` library. I wanted to see if I could use that framework to implement teleop on the **Unitree Go2**, just for my learning.

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

#### PD Control Law
The torque applied to each joint follows a standard PD controller:
$$
\tau_i = K_p (q_{target,i} - q_i) + K_d (\dot{q}_{target,i} - \dot{q}_i)
$$
where:
* $K_p = 20$ is the proportional gain
* $K_d = 0.5$ is the derivative gain
* $q_{target,i}$ is the target joint position from the policy
* $q_i, \dot{q}_i$ are the current joint position and velocity

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
$$
q_{target} = q_{default} + 0.25 \cdot \text{Action}
$$

To bridge the "sim-to-real" gap, I implemented a **one-step action delay**. The action computed at time $t$ is not applied until $t+1$, mimicking the communication latency of real hardware.

---

## 4. Rewards

### 4.1 Tracking Rewards
The primary objective is to follow the user's command.

#### Linear Velocity Tracking
$$
r_{lin} = w_{lin} \cdot \exp\left(-\frac{\|\mathbf{v}_{cmd} - \mathbf{v}_{actual}\|^2}{\sigma_{lin}^2}\right)
$$
where $w_{lin} = 1.0$ and $\sigma_{lin} = 0.25$.

#### Angular Velocity Tracking
$$
r_{ang} = w_{ang} \cdot \exp\left(-\frac{(\omega_{z,cmd} - \omega_{z,actual})^2}{\sigma_{ang}^2}\right)
$$
where $w_{ang} = 0.5$ and $\sigma_{ang} = 0.25$.

#### Height Tracking
$$
r_{height} = w_{height} \cdot \exp\left(-\frac{(h_{cmd} - h_{actual})^2}{\sigma_{h}^2}\right)
$$
where $w_{height} = 0.3$ and $\sigma_{h} = 0.1$.

### 4.2 Survival & Style Penalties
To prevent the robot from flailing or damaging itself, I added:

#### Action Smoothness
$$
r_{smooth} = -w_{smooth} \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2
$$
where $w_{smooth} = 0.01$. This penalizes jerky motions.

#### Nominal Pose Regularization
$$
r_{pose} = -w_{pose} \cdot \sum_{i=1}^{12} (q_i - q_{default,i})^2
$$
where $w_{pose} = 0.1$. This keeps the robot near its natural standing configuration.

#### Vertical Stability (Non-Jump)
$$
r_{vert} = -w_{vert} \cdot v_z^2 \quad \text{(when not jumping)}
$$
where $w_{vert} = 0.5$ and $v_z$ is the vertical velocity of the base.

#### Joint Torque Penalty
$$
r_{torque} = -w_{torque} \cdot \sum_{i=1}^{12} \tau_i^2
$$
where $w_{torque} = 0.0001$. This encourages energy-efficient gaits.

#### Base Orientation Penalty
$$
r_{orient} = -w_{orient} \cdot (\text{roll}^2 + \text{pitch}^2)
$$
where $w_{orient} = 0.2$. This keeps the body level.

### 4.3 The Jump Logic
Jumping is handled via a Finite State Machine (FSM) inside the reward function. When the "jump" command is active:

#### State 1: Preparation (Crouch)
$$
r_{prep} = -0.1 \cdot r_{pose} - 0.1 \cdot r_{orient}
$$
Penalties are relaxed by 90% to allow the robot to crouch.

#### State 2: Takeoff (Peak)
$$
r_{takeoff} = w_{jump\_height} \cdot \exp\left(-\frac{(h_{target} - h_{actual})^2}{\sigma_{jump}^2}\right) + w_{jump\_vel} \cdot \mathbb{1}_{v_z > 0} \cdot v_z
$$
where:
* $w_{jump\_height} = 500.0$ (massive reward for reaching target height)
* $w_{jump\_vel} = 100.0$ (reward upward velocity)
* $h_{target}$ is the commanded jump height
* $\mathbb{1}_{v_z > 0}$ is an indicator function (1 if $v_z > 0$, else 0)

#### State 3: Landing
$$
r_{land} = w_{land} \cdot \exp(-\|\mathbf{v}_{base}\|^2 / \sigma_{land}^2) - 2.0 \cdot r_{orient}
$$
where $w_{land} = 10.0$ and $\sigma_{land} = 0.5$. This rewards soft landings and penalizes tipping over.

#### Total Reward
$$
R_t = \sum_{i} r_i(s_t, a_t)
$$
The final reward is the sum of all active reward components at each timestep.

---

## 5. Training with PPO

I trained the policy using Proximal Policy Optimization (PPO) provided by the **RSL-RL** library. The actor and critic networks are simple Multi-Layer Perceptrons (MLP) with ELU activations:
`Input (48) -> 512 -> 256 -> 128 -> Output`

### 5.1 PPO Objective Function

PPO optimizes a clipped surrogate objective to prevent destructively large policy updates:

#### Clipped Surrogate Loss
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where:
* $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio
* $\hat{A}_t$ is the estimated advantage (see GAE below)
* $\epsilon = 0.2$ is the clipping parameter

#### Value Function Loss
$$
L^{VF}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - R_t)^2 \right]
$$

where $V_\phi(s_t)$ is the critic's value estimate and $R_t$ is the empirical return.

#### Entropy Bonus
$$
L^{ENT}(\theta) = \mathbb{E}_t \left[ H(\pi_\theta(\cdot | s_t)) \right] = \mathbb{E}_t \left[ -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]
$$

where $c_{ent} = 0.01$ is the entropy coefficient, encouraging exploration.

#### Total Loss
$$
L_{total} = -L^{CLIP}(\theta) + c_{vf} \cdot L^{VF}(\phi) - c_{ent} \cdot L^{ENT}(\theta)
$$

where $c_{vf} = 0.5$ is the value function coefficient.

### 5.2 Generalized Advantage Estimation (GAE)

Advantages are computed using GAE-$\lambda$ for bias-variance tradeoff:
$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where the TD-error is:
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Parameters:
* $\gamma = 0.99$ (discount factor)
* $\lambda = 0.95$ (GAE parameter)

### 5.3 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | $3 \times 10^{-4}$ | Adam optimizer |
| **Num Epochs** | 5 | PPO updates per batch |
| **Mini-batch Size** | 204,800 | $= 4096 \times 50$ steps |
| **Clip Range** | 0.2 | PPO clipping $\epsilon$ |
| **Max Grad Norm** | 1.0 | Gradient clipping |
| **Environments** | 4096 | Parallel simulations |
| **Horizon** | 50 | Steps before update |
| **Iterations** | 1500 | Total training iterations |

**Training Duration:** Approximately 50 minutes on an RTX 4050.

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