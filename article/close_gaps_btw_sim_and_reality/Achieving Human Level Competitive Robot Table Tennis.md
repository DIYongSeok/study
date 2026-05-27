## Executive Summary

This paper presents the first learned robotic agent to achieve **amateur human-level performance** in competitive table tennis. Developed by Google DeepMind, the system utilizes a hierarchical and modular policy architecture trained entirely in simulation and successfully transferred zero-shot to the physical world. In a user study involving 29 human players spanning different skill levels, the robot won **45% of its matches** ($13/29$), achieving a $100\%$ win rate against beginners and a $55\%$ win rate against intermediate players.

---

## Technical Architecture

To handle the high speed, precise control, and strategic decision-making required for table tennis, the authors developed a **hierarchical and modular system**:

* 
**Low-Level Controllers (LLCs):** A library of 17 specialized skill policies (e.g., forehand topspin, backhand targeting, underspin serves). These policies process 8 timesteps of ball and robot history to output joint velocity commands at $50\text{ Hz}$. They were trained via an evolutionary strategies algorithm called **Blackbox Gradient Sensing (BGS)**, chosen because it produces smoother actions than standard RL algorithms like PPO or SAC.


* 
**High-Level Controller (HLC):** Triggered once per ball hit, the HLC strategic module determines the optimal style (forehand or backhand) and switches between specialized rallying or serving LLCs. The control flow operates within $20\text{ ms}$.


* 
**Skill Descriptors:** To help the HLC make strategic choices, each LLC is mapped onto a **KD-Tree lookup table**. This allows the robot to query its own estimated landing rate, landing location, and hit velocity for any incoming ball state.



---

## Sim-to-Real Paradigm

A core contribution of the paper is bridging the physical and task-distribution sim-to-real gaps without time-consuming real-world fine-tuning:

1. 
**Iterative Real-World Grounding:** Instead of sampling uniformly, the simulator directly samples from a non-parametric dataset of real human ball trajectories. Gaps in the robot's capabilities were iteratively collected across **7 evaluation cycles with over 50 human opponents**, building an automatic task curriculum over 3 months.


2. 
**Spin Correction and FiLM Layers:** Due to the highly nonlinear and bimodal nature of paddle-rubber contact parameters between topspin and underspin, directly deployed policies failed on topspin. The authors integrated task-conditioned physics parameters in MuJoCo and trained thin **Feature Lineasized Embedding (FiLM) adapter layers** to close the remaining gap.


3. 
**Real-Time Online Adaptation:** During live matches, a simple gradient bandit algorithm tracks match statistics and updates **numerical preferences ($H$-values)** for individual LLCs. This allows the robot to adjust to a specific opponent's strengths or weaknesses in real-time.



---

## Experimental Setup & Hardware

* 
**Robot Embodiment:** A 6 Degree-of-Freedom (DoF) ABB IRB 1100 robotic arm mounted on two perpendicular Festo linear gantries ($4\text{ m}$ side-to-side, $2\text{ m}$ forward-and-back) providing rapid 2D planar mobility.


* 
**Perception:** A pair of Ximea cameras tracking the ball at $125\text{ Hz}$ coupled with a PhaseSpace motion capture system (20 cameras) tracking the human opponent's custom LED-equipped paddle.


* 
**Paddle Construction:** A 3D-printed handle equipped with **short pips rubber**.



---

## Key Results and Findings

### User Study Performance

The full competitive matches against 29 unseen human players resulted in a clear skill-level boundary:

| Opponent Skill Level | Match Win Rate | Points Won % |
| --- | --- | --- |
| <br>**Beginner** 

 | <br>$100\%$ 

 | <br>$72\%$ 

 |
| <br>**Intermediate** 

 | <br>$55\%$ 

 | <br>$50\%$ 

 |
| <br>**Advanced / Advanced+** 

 | <br>$0\%$ 

 | <br>$34\%$ 

 |

### Human-Robot Interaction (HRI) Insights

* 
**Lasting Appeal:** On a 5-point Likert scale, participants gave an average rating of **4.87** for their interest in playing with the robot again, with many calling it "dynamic," "fun," and "exciting".


* 
**Psychological Edge:** The robot consistently won the first game against beginner and intermediate players. Survey feedback revealed that the human players found the high-speed robot intimidating and loud initially before adjusting in games 2 and 3.



---

## Known Limitations

Despite achieving amateur-level play, the authors highlighted several areas that advanced players successfully exploited:

1. 
**Heavy Underspin & Low Balls:** Because of a strict collision-avoidance protocol preventing the paddle from hitting the table, the robot could not effectively return low balls, dropping its return rate drastically against heavy underspin.


2. 
**High Latency:** System latency (~$100\text{ ms}$) and the hard requirement to reset to a fixed pose between hits restricted its ability to cope with very fast balls.


3. 
**Lack of Long-Term Strategy:** The system functions on a "one ball at a time" basis, lacking multi-ball strategic planning.


4. 
**Hardware Blind Spots:** The robot cannot handle balls higher than $\sim 6\text{ ft}$ (due to camera FOV limits) or balls dropping too close to the net.