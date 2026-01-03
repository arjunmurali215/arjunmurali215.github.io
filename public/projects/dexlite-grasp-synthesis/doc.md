---
title: "DexLite: Replicating State-of-the-Art Dexterous Grasping (On a Budget)"
date: "2025-12-20"
excerpt: "A lightweight implementation of the Dex1B grasp synthesis pipeline."
coverImage: "./assets/dexlite.mp4"
---

# DexLite: Replicating State-of-the-Art Dexterous Grasping (On a Budget)

**Code:  https://github.com/arjunmurali215/dexlite**

> **Note:** This project is an implementation of the grasp synthesis methodology presented in the paper **Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation**. The neural network architecture and energy functions described herein are based on their published work, adapted for consumer hardware.

**Abstract:** This post details my journey building DexLite, a learning-based system for synthesizing dexterous grasps on a Shadow Hand. By adapting the massive-scale Dex1B pipeline for a standard laptop GPU, I explore the intersection of generative deep learning and physics-based optimization.

---

## The "Fine, I'll Do It Myself" Moment

A few months ago, researchers from UC San Diego released a fascinating paper titled **Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation**. It proposed a massive-scale approach to learning dexterous manipulation, utilizing a dataset of one billion demonstrations to solve complex grasping and articulation tasks.

I read it, thought it was brilliant, and immediately went hunting for the code.
Result: **No code online.**

So I decided to implement the paper myself, atleast whatever I could in simulation. I wanted to understand the nuts and bolts of how they achieved such high-quality results by combining optimization with generative models. I call my implementation **DexLite**—a lightweight, accessible version of their massive pipeline (some corners were cut because of hardware constraints).

![Example Synthesized Grasp](./assets/dexlite.mp4)

---

## The Challenge: High-DOF Grasping

Why is this hard? Unlike a simple parallel-jaw gripper, the **Shadow Hand** possesses high degrees of freedom (DoF), making it incredibly challenging to control effectively. It is essentially a human hand.

Finding a valid configuration for these joints that results in a stable grasp is an optimization problem which is very slow for generating large datasets. The Dex1B paper solves this by identifying two key issues in generative models: feasibility (lower success rates) and diversity (tendency to interpolate rather than expand).



---

## How DexLite Works

My implementation follows the core philosophy of the paper, integrating geometric constraints into a generative model. The pipeline consists of three main stages:

1.  **The Neural Network:**
    The heart of the system is a conditional generative model. It starts with **PointNet**, which processes the object's point cloud to extract global geometric features and local features for specific surface points. These features are fed into a **Conditional Variational Autoencoder (CVAE)**.

    ![Network Architecture](./assets/architecture.jpeg)

    The CVAE structure allows for two distinct modes of operation:
    * **Dataset Expansion:** During training, we can input existing valid grasps along with the object features into the Encoder to map them to a latent space. By slightly varying the "associated point" (the target point on the object) or the latent vector, we can decode variations of known successful grasps, effectively multiplying our dataset.
    * **Pure Synthesis:** For generating completely new grasps at inference time, we ditch the Encoder entirely. We sample random noise from a standard normal distribution and feed it into the Decoder along with the object features. The Decoder then "hallucinates" a valid grasp configuration from scratch.

2.  **The Losses:**
    You can't just train this on visual similarity (MSE) alone. To ensure the generated hands don't look like spaghetti or clip through the object, I implemented a comprehensive set of energy functions:
    * **Reconstruction Loss ($L_{recon}$):** Keeps the generated grasp close to the ground truth during training.
    * **KL Divergence ($L_{KL}$):** Regularizes the latent space so we can sample from it later.
    * **Force Closure ($E_{fc}$):** Ensures the grasp is physically stable and resists external wrenches.
    * **Penetration Penalty ($E_{sdf}$):** Uses Signed Distance Functions (SDF) to punish fingers for clipping inside the object mesh.
    * **Contact Distance ($E_{dis}$):** Acts as a magnet, pulling fingertips towards the object surface to ensure contact.
    * **Self-Penetration ($E_{spen}$):** Prevents the hand from colliding with itself.
    * **Joint Limits ($E_{joints}$):** Ensures the hand doesn't bend in physically impossible ways.

3.  **Post-Optimization:**
    A final optimization step that fine-tunes the fingers to ensure solid contact, minimizing the energy function $E_{post}$.

---

## Key Implementation Differences (The "Lite" Part)

The original Dex1B pipeline is designed to generate **one billion** demonstrations using massive compute clusters. My constraints were slightly different: I am running this on a laptop with an **RTX 4050**.

To make this feasible, I had to be smarter about my data:

* **Curated Data vs. Raw Generation:** The paper generates a seed dataset of ~5 million poses using pure optimization. Instead of burning my GPU for weeks, I curated a high-quality subset from the existing **DexGraspNet** dataset.
* **Rigorous Filtering:** I built a custom validation pipeline using PyBullet and MuJoCo. I ran stability tests (does the object sit on the table?) and lift tests (does the grasp actually hold?) before training. This reduced the dataset size but drastically increased quality.

---

## The Secret Sauce: Post-Optimization

Here was a big takeaway from this project: NO matter what, **The neural network is not enough.**

The raw output from the CVAE is good, but often suffers from lower success rates than deterministic models. It gets close to a successful grasp, but doesn't achieve it. Maybe it penetrates, or maybe it does not make contact. But tiny adjustments can make it successful.

I implemented the post-optimization step suggested in the paper. It takes the sampled hand poses and refines them to prevent penetration and ensure the fingers cover the object closely.

* **Raw Network Output:** ~55% Success Rate (Grasps often loose or clipping).
* **With Post-Optimization:** ~79% Success Rate (Tight, physically valid grasps).

As the paper notes, this hybrid approach leverages the best of both worlds: optimization ensures physical plausibility, while the generative model enables efficiency.

---

## Conclusion & Future Work

Replicating Dex1B was a lesson in the importance of hybrid approaches. Deep learning provides the intuition, and classical control theory provides the precision.

I’m planning to extend this work by incorporating "Graspness" (learning which parts of an object are graspable) and potentially moving to dual-hand manipulation.

---

**Code:  https://github.com/arjunmurali215/dexlite**