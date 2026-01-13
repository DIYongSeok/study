# Generating Activity Snippets by Learning Human-Scene Interactions

**Source:** Changyang Li and Lap-Fai Yu. 2023. ACM Trans. Graph. (SIGGRAPH) [cite: 28]

## 1. Overview
This paper presents a novel approach to generating **"Activity Snippets,"** defined as sequences of keyframes that depict multi-character, multi-object interactions in 3D environments[cite: 17, 64]. The goal is to synthesize plausible virtual activities (e.g., a waiter taking an order from customers) by learning from recordings of real-world human-scene interactions[cite: 19].

The system addresses the demand for generating complex background behaviors for Non-Player Characters (NPCs) in the metaverse, games, and mixed reality, moving beyond static poses to dynamic, high-level interaction planning[cite: 39, 44, 92].

---

## 2. Core Concept: Activity Snippet Representation
An activity snippet is represented as a sequence of keyframes, where each keyframe contains two distinct layers of information:

1.  **Keyframe Description ($G_k$):** A graph representing abstract interactions[cite: 145].
    * **Nodes:** Represents instances, including characters (e.g., waiter) and objects (e.g., table, plate)[cite: 48, 152].
    * **Edges:** Encodes semantic relationships, such as "look at," "hold," "sit on," or "side"[cite: 48, 151].
2.  **3D Placements ($\Phi_k$):** The parametric 3D poses of the instances[cite: 146].
    * **Objects:** Parameterized by position and orientation[cite: 219].
    * **Characters:** Parameterized by position, body orientation, and head orientation[cite: 220].

---

## 3. Algorithm Explanation
The proposed method operates in two distinct stages: **Generation** (creating the interaction graphs) and **Instantiation** (optimizing the 3D scene).

### Stage 1: Keyframe Description Generation
The system generates a sequence of graphs ($G_1, G_2, \dots$) using a **sequential deep graph generative model**[cite: 20].

* **Sequential Generation Logic:** The generator produces the next keyframe description ($G_{k+1}$) conditioned on the previous graph ($G_k$) and a temporal signal[cite: 56].
* **Temporal Module:** A recurrent neural network (LSTM) functions as a temporal module ($f_{temp}$)[cite: 139]. It takes a graph summary and the previous state to output a generation signal ($s^{k+1}$) and decides when the activity should terminate via a termination function ($f_{termin}$)[cite: 57, 139, 140].
* **Graph Modification Cycle:** To transition from frame $k$ to $k+1$, the generator modifies the graph through four sequential modules[cite: 292, 312]:
    1.  **Delete Nodes:** Decides if instances (e.g., a customer leaving) should be removed[cite: 293].
    2.  **Delete Edges:** Removes outdated interactions[cite: 297].
    3.  **Add Nodes:** Introduces new characters or objects into the scene[cite: 301].
    4.  **Add Edges:** Establishes new interactions (e.g., a character starts holding an object)[cite: 304].
* **Message Propagation:** The model uses message-passing graph convolution to update node features based on neighbors and edge categories[cite: 124, 272].

### Stage 2: Activity Snippet Instantiation
Once the abstract graphs are generated, the system computes the 3D poses. Since the solution space is complex, the authors devise a **two-stage optimization framework** utilizing Simulated Annealing[cite: 21, 58, 428].

#### A. Constraint Formulation
Edges in the graph are converted into cost functions for optimization[cite: 352]:
* **Position-based ($C_p$):** Encodes spatial relations. For example, "on the side of" requires the distance between nodes to be less than a target threshold $D$ (e.g., 0.4m)[cite: 358, 367].
* **Orientation-based ($C_o$):** Encodes facing directions. For example, "look at" minimizes the angle between a character's head direction and the vector pointing to the target[cite: 376, 394].
* **Attachment:** Hard constraints (e.g., "sit on") enforcing that a child instance moves consistently with its parent instance[cite: 400].

#### B. Coarse 2D Optimization (Phase 1)
* **Process:** The 3D scene is projected onto a 2D plane[cite: 425].
* **Objective:** Optimizes only **position-based** constraints to find a rough initialization[cite: 425].
* **Progressive Schema:** To avoid local minima, the optimizer uses a relaxation factor $\lambda$. It starts with relaxed constraints (allowing larger distances) and progressively tightens them until $\lambda=1$[cite: 453, 475].

#### C. Fine 3D Optimization (Phase 2)
* **Process:** Uses the output of the 2D stage as initialization and expands to full 3D space[cite: 427].
* **Objective:** Optimizes **both** position and orientation constraints[cite: 491].
* **Refinement:** Because positions are already roughly accurate, the solver prioritizes rotation updates (probability 0.8) over position updates (probability 0.2)[cite: 492].

---

## 4. Key Results & Applications
* **Plausibility:** A perceptual study confirmed that generated snippets were rated as reasonable and plausible by human evaluators[cite: 24, 910].
* **Interactive Generation:** The system can interactively generate snippets where a human player controls one character, and the model generates the behaviors of the surrounding characters and objects in response[cite: 60, 816].
* **Mixed Reality:** The framework can augment real-world scenes (e.g., a real table scanned via HoloLens) with virtual characters performing context-aware activities[cite: 889, 922].