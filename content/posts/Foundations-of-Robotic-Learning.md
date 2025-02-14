---
date: '2025-02-08T18:25:16Z'
draft: false
title: 'Robotic Learning Part 1: The Physical Reality of Robotic Learning'
author: 'Alexander Quessy'
ShowReadingTime: true
math: true
diagram: true
---

To understand why robot learning is fundamentally different from traditional machine learning, let's start with a simple example. Imagine teaching a robot to pick up a coffee cup. While a computer vision system needs only to identify the cup in an image, a robot must answer a series of increasingly complex questions: Where exactly is the cup? How should I move to grasp it? How hard should I grip it? What if it's fuller or emptier than expected?

This seemingly simple task illustrates why robot learning isn't just about making predictions, it's about making decisions that have physical consequences.

## Sequential Decision Making Under Uncertainty

In traditional ML, each prediction stands alone. If a model misclassifies an image, the next image it sees is completely unaffected. But robots operate sequentially, where each action changes the world in ways that affect all future decisions. We can describe this mathematically as a sequence:
$$
\tau = (s_{0}​,a_{0}​,s_{1}​,a_{1}​,...,s_{T}​)
$$
where $s_{t}$ represents the state at time $t$ (like the position of the gripper and cup) and $a_{t}$ represents the action taken (like moving the gripper). Each action doesn't just affect the immediate next state action, it can influence the entire future trajectory of the task.

{{< figure src="/Gripper500.gif" alt="☕️ Gripper" >}}

This sequential decision making process is made even more challenging by the fact that robots must deal with uncertainty. These can be generally classified into 3 different types of uncertainty: 

1. **Perception Uncertainty**: When a robot observes the world through its sensors, what it sees is incomplete and noisy. Mathematically this can be written as $o_{t} = s_{t} + \epsilon$ where  $s_{t}$ is what the robot should ideally observe, and $\epsilon$ represents noise. Real robots generally combine multiple sensors, each with their own challenges. Examples include:
	- [**Cameras**](https://thepihut.com/products/12mp-imx477-mini-hq-camera-module-for-raspberry-pi?variant=32522522951742&country=GB&currency=GBP&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&gad_source=1&gbraid=0AAAAADfQ4GFSimkynGqEbghBD6YH13FKL&gclid=EAIaIQobChMIi-6Oid22iwMVNoBQBh0DhB-MEAQYASABEgLt5vD_BwE), provide dense visual information. Computer vision deriving meaningful from digital images is an entire field in itself. In robotics we are usually concerned with any problem that causes the meaning of the image to be distorted, this could be visual occlusions, changes in lighting or changes to the key visual characteristics of the scene. 
	- [**Depth Sensors**](https://www.intelrealsense.com/compare-depth-cameras/), measure the distance between to surfaces in a scene. They suffer from similar errors as cameras but are especially susceptible to errors from reflective surfaces and often struggle to detect small objects.
	- [**Force Sensors**](https://www.ati-ia.com/products/ft/ft_models.aspx?id=mini45), measure contact forces. These generally suffer from errors in calibration, either from misalignment or incorrect zero-ing of the force sensor.
	- [**Joint Sensors**](https://netzerprecision.com/products/?utm_source=google&utm_medium=cpc&utm_campaign=Search&utm_content=general&utm_term=hollow%20shaft%20absolute%20encoder&utm_campaign=Netzer+-+Search+General&utm_source=adwords&utm_medium=ppc&hsa_acc=9965475607&hsa_cam=20943823131&hsa_grp=157677495053&hsa_ad=687969984958&hsa_src=g&hsa_tgt=kwd-335054323609&hsa_kw=hollow%20shaft%20absolute%20encoder&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gad_source=1&gbraid=0AAAAACOuESPwjVV6QTy4BXvPT3T52bMk_&gclid=EAIaIQobChMI7Yzi3t22iwMVp5NQBh1QVAC6EAAYASAAEgJXqfD_BwE), measure joint angle or position. Similar to force sensors they are susceptible to errors in calibration and alignment. 
	
	Putting it all together Boston Dynamic's Humanoid Atlas Robot has 40-50 sensors, as you can imagine this means there is a lot of uncertainty they need to deal with in order to understand the state of the robot.
{{< webm-video file="blogs/Foundations-of-Robotic-Learning/perception-atlas.webm" >}}

1. **Action Uncertainty**: Even when a robot knows how to behave, executing that action perfectly is impossible. For example in the simple coffee cup picking task there is still noise from mechanic imperfections, changes in motor temperature, latency in the control system, robotic wear and tear over time.

2. **Environment Uncertainty**: The real world is messy and unpredictable. Physical properties can significantly vary the the way the robot needs to behave in our example:
	- The material the cup is made from could deform or be slippery
	- The cup could have a different mass than expected
	- The cup may not be where we expected it to be on the table

Putting this all together, our robotic cup picking up algorithm needs to handle the following functions, each with its own sources of accumulating uncertainty:

```python
def pick_up_cup():
	
	cup_position = get_cup_position()  # Perception
	planned_path = plan_motion(cup_position)  # Planning
	actual_motion = execute_path(planned_path)  # Control
	contact_result = grip_cup()  # Sensing
	
	return contact_result
```

This is why robotic learning algorithms need expertise that regular ML algorithms don't:
1. They must be robust to noise
2. The need to handle partial and imperfect information
3. They must adapt to changing conditions
4. They need to be cautious when uncertainty is high

## Linking Perception to Action

At its core robot learning requires 3 key components:
- A way to perceive the world
- A way to decide what to do
- A way to execute that action
With this in mind we can build a general model to account for each of these components.
### State Space 

A robot's state space represents everything we can observe in the environment for the coffee picking robot this might include:

```python
state = {
	'joint_positions': [1.2, -0.5, 1.8],        # Where are my joints?
	'joint_velocities': [0.115, 0.00, -0.211],  # How fast are they moving?
	'camera_image': np.array([...]),            # What do I see?
	'force_reading': [200.1, 310.2, 0.9],       # What do I feel?
	'gripper_state': "OPEN"                     # What's the state of my hand?
}
```

These states are constantly evolving and encompass a variety of dissimilar data-types. 

### Action Space

A robot's action space defines what it can actually do in the environment this might include:

```python
action = {
	'joint_velocities' = [-0.13, 0.21, 0.55]  # How fast to move each joint
	'gripper_command' = "CLOSE"               # How to move my hand
}
```

### Control loop

Now that we understand state and action spaces, let's explore how robots use this information to actually make decisions. The key concept here is the control loop - the continuous cycle of perception and control that allows robots to interact with the world.

{{< mermaid >}}
graph LR
    A[Observe] --> B[Decide]
    B --> C[Act]
    C --> A

    style A fill:#e1f5fe,stroke:#01579b
    style B fill:#fff3e0,stroke:#e65100
    style C fill:#e8f5e9,stroke:#1b5e20
{{< /mermaid >}}

This control loop becomes far more interesting when we consider how to make decisions under uncertainty. This is where the concept of Markov Decision Processes (MDPs)[^1] become helpful. An MDP provides a mathematical framework for making sequential decisions when outcomes are uncertain. In the context of MDPs, at each time-step $t$:
- The robot finds itself in a state $s_{t}$
- It takes an action $a_{t}$, according to some policy $\pi(s_{t})$
- This leads to a new state $s_{t+1}$ with some probability $P(s_{t+1}|s_{t}, a_{t})$
- The robot receives a reward $r(s_{t}, a_{t})$

The *Markov* part of the MDP comes from a key assumption: 

> The next state depends **only** on the current state and action, **not** on the history of how we got here. 

Let's unpack what this means for our coffee cup picking robot.

Imagine our gripper is hovering $10cm$ above the cup. According to the Markov property to predict what happens when we move down $2cm$, we *only* need to know:
- Current state ($10 cm$ above the cup)
- Current action (move down $2cm$)
- Current sensor readings (force, vision, etc)

It doesn't matter how we got to this position, whether we just started the task, or if we have been trying for hours, or whether we previously dropped the cup. The trick is that the state needs to include all information that is important to make decisions. So if the number of times we dropped the cup is important to the decisions we make it should be included in our state. 

This turns out to be very helpful. By carefully choosing what information to include in our state, we can capture all relevant history while keeping our problem definition simple and tractable. 

## Why this matters for Robotic Learning?

The MDP framework is especially useful for Robotic learning for three key reasons:
1. **Uncertainty**: MDPs model probabilities explicitly. When grasping a cup, we can express that: "closing the gripper has an 80% chance of secure grasp, 15% chance of partial grip, and 5% chance of missing entirely."
2. **Long-term consequences**: Small errors compound over time. For example, a $1cm$ misalignment during grasping might let us pick up the cup, but could lead to spilling during transport. The MDP framework captures this through its reward structure and state transitions, even though each state transition only depends on the current state (Markov property), the cumulative rewards over the sequence of states let us optimize for successful task completion. A spilled cup means no reward, guiding the policy toward careful movements even if the cup is slightly misaligned.
3. **Algorithm design**: The MDP framework helps shape how we think about robotic learning problems and building autonomous systems:
	- Reinforcement Learning[^2] (RL) optimises for long-term rewards across state transitions.
	- Model-Predictive Control[^3] (MPC) uses explicit models of state transitions to plan sequences of actions.
	- Imitation Learning (IL)[^4] can learn from human demonstrations by modelling them as optimal MDP solutions.

## Citation

> Quessy, Alexander. (2025). Robotic Learning for Curious People. *aos55.github.io/deltaq*. [https://aos55.github.io/deltaq/posts/an-overview-of-robotic-learning/](https://aos55.github.io/deltaq/posts/an-overview-of-robotic-learning/).

```bibtex
@article{quessy2025roboticlearning,
  title   = "Robotic Learning for Curious People",
  author  = "Quessy, Alexander",
  journal = "aos55.github.io/deltaq",
  year    = "2025",
  month   = "Feb",
  url     = "https://aos55.github.io/deltaq/posts/an-overview-of-robotic-learning/"
}
```

## References

[^1]: R. Bellman, *Dynamic Programming*. Princeton, NJ: Princeton University Press, 1957 

[^2]: R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018

[^3]: E. F. Camacho and C. Bordons, *Model Predictive Control*. London, UK: Springer, 2007.

[^4]: S. Schaal, *Is imitation learning the route to humanoid robots?*, Trends Cogn. Sci., vol. 3, no. 6, pp. 233–242, June 1999.
