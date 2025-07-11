---
title: 'Robotic Learning for Curious People'
date: '2025-02-08T16:52:38Z'
draft: false
author: "Alexander Quessy"
ShowReadingTime: true
math: true
diagram: true
toc: true
---

Robot learning combines robotics and machine learning to create systems that learn from experience, rather than following fixed programs. As automation extends into streets, warehouses, and roads, we need robots that can generalise, taking skills learned in one situation and adapting them to the countless new scenarios they'll encounter in the real world. This series explains the key ideas, challenges, and breakthroughs in robot learning, showing how researchers are teaching robots to master flexible, adaptable skills that work across the diverse and unpredictable situations of the real world.

## Introdction

In 1988, roboticist Hans Moravec made an observation: skills that humans find effortless, like [mixing a drink](https://www.youtube.com/watch?v=rDxTsjD-dKw), [making breakfast](https://www.youtube.com/watch?v=E2evC2xTNWg) or [walking on uneven ground](https://www.youtube.com/watch?v=g0TaYhjpOfo), are incredibly difficult for robots. Meanwhile, tasks we find mentally challenging, like [playing chess](https://www.chess.com/terms/alphazero-chess-engine) or [proving theorems](https://www.nature.com/articles/d41586-025-00406-7), are relatively straightforward for machines. This counterintuitive reality, known as Moravec's paradox, lies at the heart of why robot learning has become such an exciting and challenging field.

Think about a toddler learning to manipulate objects. They can quickly figure out how to pick up toys of different shapes, adapt their grip when something is heavier than expected, and learn from their mistakes. These capabilities, represent some of our most sophisticated yet often least appreciated forms of intelligence. As Moravec noted:

> We are all prodigious olympians in perceptual and motor areas, so good that we make the difficult look easy.[^1]

{{< webm-video file="blogs/An-Overview-of-Robotic-Learning/BimanualObject.webm" caption="Figure 1: A [robot placing balls](https://www.youtube.com/watch?v=9d6hiqLtml8) in a pot.">}}

{{< webm-video file="blogs/An-Overview-of-Robotic-Learning/BabySorting.webm" caption="Figure 2: A [baby placing balls](https://www.youtube.com/watch?v=pb3aUNl52oQ) in a box.">}}

This is where robot learning emerges as a compelling solution. Traditional robotics relied on carefully programmed rules and actions - imagine writing specific instructions for every way a robot might need to grasp different objects. This approach breaks down in the real world, where even slight variations in lighting, object position, or surface texture can confuse these rigid systems. A robot programmed to pick up a specific coffee mug might fail entirely when presented with a slightly different one.

Robot learning offers a fundamentally different approach. Instead of trying to anticipate and program for every possible scenario, we let robots discover solutions through experience and adaptation. Just as a child learns to grasp objects through trial and error, modern robots can learn from their successes and failures, gradually building up robust behaviours that work across diverse situations.

## Prerequisites
To understand the approaches we'll discuss, you should have:

- Good understanding of [probability](https://greenteapress.com/wp/think-stats-3e/) and [linear algebra](https://minireference.com/static/tutorials/linear_algebra_in_4_pages.pdf).
- Basic familiarity with [machine learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) and [deep learning](https://www.deeplearningbook.org).
- [Basic](https://allendowney.github.io/ThinkPython/chap00.html) [programming](https://docs.fast.ai) and [computer](https://mimoza.marmara.edu.tr/~msakalli/cse706_12/SkienaTheAlgorithmDesignManual.pdf) [science](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html) knowledge. 
- Basic understanding of [robotics](https://marsuniversity.github.io/ece387/Introduction-to-Robotics-Craig.pdf)/[mechaniscs](https://oxvard.wordpress.com/wp-content/uploads/2018/05/engineering-mechanics-dynamics-7th-edition-j-l-meriam-l-g-kraige.pdf) and [control](https://ctms.engin.umich.edu/CTMS/index.php?aux=Home).

## What These Posts Cover
We'll explore how robot learning is tackling Moravec's paradox:

1. [The Fundamentals](https://aos55.github.io/deltaq/posts/foundations-of-robotic-learning/): Why *simple* robotic tasks are actually complex.
2. [Learning Paradigms](https://aos55.github.io/deltaq/posts/key-learning-paradigms-in-robotics/): How to teach robots through demonstrations and experience.
3. [The Reality Gap](https://aos55.github.io/deltaq/posts/the-reality-gap/): Why simulation alone isn't enough, and what we can do about it.
4. [Modern Approaches](https://aos55.github.io/deltaq/posts/modern-approaches/): How new techniques are making headway on these problems.
5. Real World Applications: How these techniques are being applied in the real-world.

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

[^1]: Minsky, M. (1988). *The Society of Mind*. New York: Simon and Schuster.
