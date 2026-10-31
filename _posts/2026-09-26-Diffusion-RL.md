---
layout: blog_post
title: 'Learning and Thinking about Diffusion RL'
date: 2026-09-26
tags:
  - Diffusion
  - RL
  - Article
---

This is still a developing draft. But welcome to take a look anyway.

<!-- {% capture math_content %} -->

## 0. Foreword

This blog is largely inspired by [**Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View**](https://www.arxiv.org/pdf/2608.14430) (NeurIPS 2026), which I might casually abbreviate to '0814' later. I would say it led me to actively exploring the field of Diffusion RL.

> Why write this blog?

I was amazed by [**DiffusionNFT**](https://arxiv.org/abs/2509.16117) for its clean formulation and how a policy is dragged linearly towards a preferred distribution in its training process. However as I search for surveys on RL for post-training diffusion models, most papers I get are either [outdated surveys](https://arxiv.org/abs/2311.01223) not covering these methods, or surveys about how DM plays a role in RL - not at all relevant to my curiosity. Why can one diffusion RL algorithm look like REINFORCE on an SDE trajectory, while another looks like supervised regression on analytically noised endpoints and both are called *'RL'*? And that's why 0814 felt so inspiring - it answers that question gracefully, placing recent methods together in one paper and providing an insightful explanation for viewing them. **In this blog I try to write down my thoughts while learning about 0814 and exploring neighboring methods not covered in this paper**. (Disclaimer: This is not a formal survey of any kind. I will only draw comparisons between methods that intrigued me.)

(10/18/26: The topic of this survey is probably what I am looking for, but it is, from my point of view, not very insightful. [Diffusion RL Survey Oct. 15th](www.arxiv.org/abs/2610.21539))

Useful links:
+ [Flow-Factory (easy T2I RL training)](https://github.com/X-GenGroup/Flow-Factory)
+ [Awesome-Visual-Generation-Alignment-Survey](https://github.com/XueZeyue/Awesome-Visual-Generation-Alignment-Survey) (Outdated, updated Nov. 2025)
+ [Awesome Diffusion Flow RL](https://github.com/AquaHorseM/Awesome-Diffusion-Flow-RL)


> Errors are possible. I apologize for possible mistakes and welcome any form of comments / corrections / suggestions.

## 1. Diffusion's curse and blessing
### 1.1 Structural obstacle of Diffusion RL

We know the canonical regularized RL objective:

$$
J(\theta) = \mathbb{E}_{P_\theta} [A] - \beta KL(P_\theta  \|  P_\text{ref}), \tag{1}
$$

where the advantage $A$ is determined solely by the terminal clean image $x_0$. Normally for PPO (in LLMs) we would like to estimate the likelihood $\nabla{\log{P_\theta(x_0)}}$, but **diffusion never provides such a convenience**. For a clean sample $x_0$ we have innumerable paths to reach it, namely calculating the marginal: $P_\theta(x_0) = \int{P_\theta(x_{[0:1]}) \mathcal{D}x_{[0:1]}}$. Over continuous paths that integral is intractable and therefore the endpoint score function $\nabla{\log{P_\theta(x_0)}}$ cannot be estimated reliably.

### 1.2 General solution: path-space

We need to calculate $\mathbb{E}_{x_0 \sim P_\theta(x_0)}[R(x_0)]$ but that marginal distribution has been proven intractable. To solve this issue a nice observation would be:

$$
\mathbb{E}_{p(x)} A(x) = \mathbb {E}_{p(x,z)} A(x) \tag{2}
$$

for whatever the latent variable $z$ (in our case this $(x,z)$ is the trajectory $\tau$)is. Eq. 2 is useful for us because we know the trajectory distribution through the product of conditional probabilities. For convenience, I will present the discrete timestep version of the trajectory (and leave continuous notations later).

$$
P_\theta(\tau) = p_1(x_1) \prod_t p_\theta (x_{t-\Delta t}  \mid  x_t) \tag{3}
$$


where $\tau$ denotes the denoising trajectory. The observation of Eq. 2 is itself very straightforward, but this probably solves half of the problem because we can now replace the intractable endpoint marginal with the entire path probability.

For an RL setting we have trajectories collected from an older policy $Q$, but again, what we need to calculate is $\mathbb{E}_{x_0 \sim P_\theta(x_0)}[R(x_0)]$ or in path-space view $\mathbb{E}_{\tau \sim P_\theta(\tau)}[A(x_0(\tau))]$, something sampled under $P_\theta$ not $Q$. Luckily, it's easy to derive:

$$
\mathbb{E}_{P_\theta(\tau)} A(x_0(\tau)) = \mathbb{E}_{Q(\tau)}{[\frac{P_\theta(\tau)}{Q(\tau)}A(x_0(\tau))]} = \mathbb{E}_{Q(\tau)} [\prod_t \frac{p_\theta (x_{t-\Delta t}  \mid  x_t)}{q(x_{t-\Delta t}  \mid  x_t)} A(x_0(\tau))] \tag{4}
$$

(The initial noise distribution $p_1(x_1)$ cancels out in $P_\theta$ and $Q$. **This is simply importance sampling and the direct source of Flow-GRPO's per step optimization.** But let's dive deeper with 0814 before returning to any specific method.)

And now it's time to replace the abstract $p$'s and $q$'s with continuous diffusion SDEs. Note that as $Q$ is an older version of the model, the stochasticity schedule stays fixed throughout the training as a selected implementation design before any training occurs at all. Thus we can assert that the diffusion coefficient $g$'s are the same for $P$ and $Q$.

**For convenience in discussing Ito calculus and avoiding $\mathrm{d}t<0$ problems, I follow the conventions of Appendix C of 0814, where $r := 1-t$ and $y_r = x_t$** Recall the target form; under this $r$ convention it would be:

$$
\frac{p_\theta (y_{r+\Delta r}  \mid  y_r)}{q(y_{r+\Delta r}  \mid  y_r)}.
$$

Schematically for $P$ and $Q$ **sampling**(i.e. reverse, i.e. $\mathrm{d}t < 0, \mathrm{d}r>0$) trajectories we have different drifts $\mu$ and shared diffusion $g$ (the forward and reverse SDEs share the same diffusion coefficient, and for simplicity $g_r := g_t$):

$$
P_\theta: \text{d}y_r = \mu_\theta(y_r, r) \text{d}r + g_r \text{d} W_r
$$

$$
Q: \text{d}y_r = \mu_q (y_r, r) \text{d}r + g_r \text{d} W_r. \tag{5}
$$

Now every discrete transition can be written as a tiny Gaussian.

$$
p_\theta(y_{r + \Delta r} \mid y_r) = \mathcal{N}(y_r + \mu_\theta\Delta r, g_r^2\Delta r)
$$

$$
q(y_{r + \Delta r} \mid y_r) = \mathcal{N}(y_r + \mu_q\Delta r, g_r^2\Delta r). \tag{6}
$$

Recall the trajectories we have now are **all sampled from $Q$**. So:

$$
y_{r+\Delta r} - y_r - \mu_q \Delta r = g_r \sqrt{\Delta r} \epsilon_r. \tag{7}
$$

Therefore,

$$
y_{r+\Delta r} - y_r - \mu_\theta \Delta r = g_r \sqrt{\Delta r} \epsilon_r - \Delta \mu(y_r, r)\Delta r. \tag{8}
$$

where $\Delta \mu := \mu_\theta - \mu_q$, simply how far is the current drift away from the older policy that collected those samples. With these identities (Eq. 6,7&8), now we obtain the desired form perfectly.

$$
\begin{align}
\log \frac{p_\theta (y_{r+\Delta r}  \mid  y_r)}{q(y_{r+\Delta r}  \mid  y_r)} & = - \frac{1}{2g_r^2 \Delta r} ( \| y_{r+\Delta r}-y_r-\mu_\theta \Delta r \| ^2 -  \| y_{r+\Delta r}-y_r-\mu_q \Delta r \| ^2) \\
& =  - \frac{1}{2g_r^2 \Delta r} ( \| g_r \sqrt{\Delta r} \epsilon_r - \Delta \mu \Delta r  \| ^2 -  \| g_r \sqrt{\Delta r} \epsilon_r \| ^2) \\
& = \frac{\Delta \mu}{g_r}  \sqrt{\Delta r} \epsilon_r - \frac{1}{2}  \| \frac{\Delta \mu}{g_r} \| ^2 \Delta r.
\end{align}
\tag{9}
$$

And this is the equation that perhaps supported the whole analysis of 0814. **A log policy ratio can be understood as linear drift-noise correlation** (Change in SDE drift aligns with the noises along that sampled good trajectory or not), **minus a quadratic energy drift cost** (How informative was the proposal $Q$?).

This log ratio itself is informative enough, but to respect the original form in 0814, it's not difficult to write out a(n) (informal) derivation. Recall Eq. 4. where we wrote (changed to the sampling path notation):

$$
\prod_r \frac{p_\theta (y_{r+\Delta r}  \mid  y_r)}{q(y_{r+\Delta r}  \mid  y_r)}
$$

From Eq. 9 we obtain:

$$
\begin{align}
\log \frac{P_\theta(\tau)}{Q(\tau)} &= \log \prod_r \frac{p_\theta (y_{r+\Delta r}  \mid  y_r)}{q(y_{r+\Delta r}  \mid  y_r)} \\
&= \sum_r \log \frac{p_\theta (y_{r+\Delta r}  \mid  y_r)}{q(y_{r+\Delta r}  \mid  y_r)}\\
&= \sum_r (\frac{\Delta \mu(y_r,r)}{g_r}  \sqrt{\Delta r} \epsilon_r - \frac{1}{2}  \| \frac{\Delta \mu(y_r,r)}{g_r} \| ^2 \Delta r)
\end{align}\tag{10}
$$

By letting $\Delta r \rightarrow 0$ we (heuristically) turn that discrete sum into a continuous integral (standard brownian motion $\mathrm{d}W_r = \epsilon_r \sqrt{\Delta r}$):

$$
\log \frac{\mathrm{d} P_\theta}{\mathrm{d} Q} = \int_0^1 (\frac{\Delta \mu(y_r,r)}{g_r} \mathrm{d}W_r - \frac{1}{2}  \| \frac{\Delta \mu(y_r,r)}{g_r} \| ^2 \mathrm{d} r) \tag{11}
$$

And Eq.11 is the Girsanov Theorem the 0814 paper is using. The discrete version already tells what we need to know (linear minus quadratic); this is simply a nicer formulation (under regular conditions).
> (I took a slightly divergent path here because I am not at all familiar with Ito calculus and never heard of Radon-Nikodym derivatives before 😿 and I think it easier to understand the optimization geometry in discrete settings)

A quick recap: substituting this continuous-time notation ([R-N derivative](https://proofwiki.org/wiki/Definition:Radon-Nikodym_Derivative)) into Eq. 4 we obtain:

$$
\mathbb{E}_{P_\theta(\tau)} A(x_0(\tau)) = \mathbb{E}_{Q(\tau)} [\frac{\mathrm{d} P_\theta}{\mathrm{d} Q}(\tau) A(y_1(\tau))] \tag{12}
$$

Expanding the R-N derivative $\frac{\mathrm{d}P}{\mathrm{d}Q}$ and **keeping only the first order $\Delta \mu$ pertubation** (*and we will return to the second order term in section 2*) yields (for notation convenience I will abbreviate $\mathbb{E}_{P_\theta(\tau)} A(x_0(\tau))$ to $J_{policy}$):

$$
\delta J_{policy}(\theta) = \mathbb{E}_Q[\int_0^1 \frac{\delta \mu_r}{g_r} \mathrm{d}W_r A(y_1)] \tag{13}
$$

This is REINFORCE in continuous time.

### 1.3 Turning terminal advantage to value function

But the thing is the advantage (or the reward) is a terminal value only determined by $y_1$ (or $x_0$ if you prefer) solely. A natural question that arises is whether the terminal advantage tells us that the Brownian movement at time $r$ (the $\mathrm{d}W_r$) is good or not?

To answer that question, it would be natural to calculate the expectation of the terminal advantage $A(y_1)$ under the condition $y_r$, i.e. the current state. We can thus define:

$$
V_r := \mathbb{E}_Q [A(y_1(\tau)) \mid y_r],
$$

interpreted as the value of the current noisy middle state $y_r$. As a conditional expectation, this $V_r$ is **the best terminal advantage estimate we have**. And upon that since the process can be read as Markovian, the current state $y_r$ is a sufficient statistic (so taking the condition $y_r$ simply equals to taking the condition $\mathcal{F}_r$). In formal maths language (martingales and filters), for $r<s$ (and naturally $\mathcal{F}_r \subset \mathcal{F}_s$):

$$
\mathbb{E}_Q[V_s \mid \mathcal{F}_r] = \mathbb{E}_Q[\mathbb{E}_Q[A(y_1) \mid \mathcal{F}_s] \mid \mathcal{F}_r]
= \mathbb{E}_Q [A(y_1) \mid \mathcal{F}_r] = V_r \tag{14}
$$

by the tower property. This is simply the formal way of saying $V_r$ is **a martingale**. Its mean is not supposed to have a drift in any direction; in other words, expressed in an SDE, its drift term **must be 0**.

Meanwhile [Ito's Lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma) tells us, with $V_r = V_r(y_r)$:

$$
\mathrm{d} V_r = (\partial_r V + \mu_Q^\top \partial_{y} V_r + \frac{1}{2}g_r^2 \partial_{yy} V_r) \mathrm{d}r + g_r \nabla_{y_r} V_r^\top \mathrm{d}W_r. \tag{15}
$$

Compounded with the fact that the drift term must be 0, we obtain:

$$
\mathrm{d} V_r = g_r \nabla V_r^\top \mathrm{d}W_r. \tag{16}
$$

From Eq. 16, we can write out the terminal advantage in another way:

$$
A(y_1) = V_1 = V_0 + \int_0^1  g_r \nabla V_r^\top \mathrm{d}W_r \tag{17}
$$

And substituting Eq. 17 into Eq. 13 yields (remember in Ito calculus we have (informally) $(\mathrm{d}W)^2 = \mathrm{d}t$, formally known as Ito isometry):

$$
\begin{align}
\delta J_{policy}(\theta) & = \mathbb{E}_Q[\int_0^1 \frac{\delta \mu_r}{g_r} \mathrm{d}W_r (V_0 + \int_0^1  g_r \nabla V_r^\top \mathrm{d}W_r)] \\
& = \mathbb{E}_Q [\int_0^1\frac{\delta \mu_r}{g_r} \mathrm{d}W_r \cdot \int_0^1  g_r \nabla V_r^\top \mathrm{d}W_r] + V_0 \cdot \mathbb{E}_Q [\int_0^1 \frac{\delta \mu_r}{g_r} \mathrm{d}W_r] \\
& = \mathbb{E}_Q [\int_0^1\nabla V_r ^ \top \delta \mu_r \mathrm{d} r] + 0.
\end{align}
\tag{18}
$$

In other words, combined with Eq. 13,

$$
\mathbb{E}_Q[\int_0^1 \frac{\delta \mu_r}{g_r} \mathrm{d}W_r A(y_1)] =
\delta J_{policy}(\theta) = \mathbb{E}_Q [\int_0^1\nabla V_r ^ \top \delta \mu_r \mathrm{d} r]
\tag{19}
$$

**We have proved that rewarding all the random Brownian movements (though noisy) along a good trajectory is, in expectation and first order to $\Delta \mu$, the same as moving the drift directly in the direction that increases expected terminal advantage.** The central amount we care about is now $\nabla V_r$.

### 1.4 Estimating the value gradient

The problem is not over yet. $\nabla V_r$ is still a hanging notation central to the RL problem. We have to **estimate** it.

One straightforward (and generic) way would be using Eq. 16: $\mathrm{d} V_r = g_r \nabla V_r^\top \mathrm{d}W_r.$ Over a small time interval $\Delta r$ we get $\Delta V_r = g_r \nabla V_r^\top \Delta W_r$, and then multiplying $\Delta W_r$ and taking a conditional expectation on both sides we get:

$$
\mathbb{E}[\Delta V_r \Delta W_r  \mid  \mathcal{F}_r] = g_r \nabla V ^ \top (\Delta W)^2 = g_r \nabla V ^ \top \Delta r \tag{20}
$$

And we observe that only the part of the terminal advantage that became predictable through this Brownian kick survives its covariance with $\Delta W_r$, since all the advantage the Brownian motion at other timesteps 'contributed' have zero conditional mean *here*.

$$
\mathbb{E}[A(y_1) \Delta W_r  \mid  \mathcal{F}_r] = \mathbb{E}[\Delta V_r \Delta W_r  \mid  \mathcal{F}_r] = g_r \nabla V ^ \top \Delta r \tag{21}
$$

Replace the expectation with an one-sample estimate and  we have an $\nabla V$ estimator, which the 0814 named as 'the stochastic estimator' because it contains the Brownian motion term $\Delta W$:

$$
\widehat{\nabla V}^\text{sto} = \frac {A(y_1) \Delta W_r}{g_r \Delta r} = \frac {A(y_1) \epsilon_r}{g_r \sqrt{\Delta r}} \tag{22}
$$

This is the general method that applies even when the stochastic dynamics is completely a black box. The general idea is simply change $x_t$ a bit, observe the terminal reward, and infer the gradient. But of course this is not the best estimate: the sampled noises are $O(1)$ yet the useful correlation is only $O(\sqrt{\Delta r})$. Dividing the noise by a small $\sqrt{\Delta r}$ significantly amplifies the variance,  i.e. $\text{Var}(\widehat{\nabla V}^\text{sto}) \sim O(\frac{1}{\Delta r})$.

And I think the following part (represented by the idea of AWM and NFT) is the real genius. We have been asking which Brownian movement turns out to lead to a great endpoint (which we said is a generic method); but diffusion provides a great convenience here. We know how $x_t$ is corrupted from $x_0$, in other words, $x_0$ exerts an analytic "pull" for the given noised state $x_t$. 'What makes a **good $x_t$, a noised middle state that is more compatible with those high-reward clean images?**'.

Recall the definition of $V_t = \mathbb{E}_Q [A(x_0) \mid x_t].$ Expanding this into integral form yields:

$$
V_t = \int A(x_0) q(x_0 \mid x_t) \mathrm{d}x_0 \tag{23}
$$

**and therefore the gradient:**

$$
\begin{align}
\nabla_{x_t}V_t &= \int A(x_0) q(x_0 \mid x_t) \nabla \log q(x_0 \mid x_t) \mathrm{d}x_0 \\
&= \mathbb{E}_Q[A(x_0)\nabla_{x_t} \log q(x_0 \mid x_t)  \mid  x_t] \\
&= \mathbb{E}_Q[A(x_0)\nabla_{x_t} (\log q(x_t \mid x_0) - \log q_t(x_t)) \mid  x_t] \\
&= \mathbb{E}_Q[A(x_0)\nabla_{x_t} \log q(x_t \mid x_0) \mid  x_t] - \mathbb{E}_Q[ \nabla_{x_t} \log q_t(x_t) \cdot A(x_0) \mid  x_t]\\
&= \mathbb{E}_Q[A(x_0)\nabla_{x_t} \log q(x_t \mid x_0) \mid  x_t] - V_t \nabla_{x_t}\log q_t(x_t)  \end{align} \tag{24}
$$

Using the mixture-score identity:

$$
\nabla_{x_t} \log q_t(x_t) = \mathbb{E} [\nabla_{x_t} \log q(x_t \mid x_0)  \mid  x_t]. \tag{25}
$$

we can obtain:

$$
\nabla_{x_t} V_t = \mathbb{E}[(A(x_0)-V_t) \nabla_{x_t} \log q(x_t \mid x_0) \mid x_t], \tag{26}
$$

which is essentially a conditional covariance between advantage and **the likelihood term measuring how compatible the noised image $x_t$ is with high-reward $x_0$ 's.** (Note the analogy with Eq. 21, which is discussing the covariance between *the advantage and a specific Brownian noise*.)

> Observation: Why don't we usually think this way as in Eq. 23 and 24? My understanding is that for a random RL setting we know nothing about the posterior $q(x_0\mid x_t)$. Diffusion does not give us the normalized posterior either, but gives us something almost as useful: an analytic derivative of the likelihood of every clean endpoint under the forward process. Bayes can then convert this into a derivative of posterior responsibility.

Returning to Eq. 26, for a rectified flow $x_t = (1-t)x_0 + t\epsilon$, $\log q(x_t \mid x_0)$ is analytical:

$$
\nabla \log q(x_t \mid x_0) = -\frac{x_t - (1-t)x_0}{t^2} \tag{27}
$$

With the definition of velocity $v = \frac{x_t-x_0}{t}$, Eq. 27 could be written as:

$$
\nabla \log q(x_t \mid x_0) = -\frac{x_t + (1-t)v}{t} \tag{28}
$$

Reusing Eq. 25 (mixture-score identity) yields:

$$
\nabla_{x_t} \log q_t(x_t) = \mathbb{E} [\nabla_{x_t} \log q(x_t \mid x_0)  \mid  x_t] = -\frac{x_t + (1-t)\mathbb{E}_Q[v \mid x_t]}{t} \tag{29}
$$

Here $\mathbb{E}_Q[v \mid x_t]$ refers to the average velocity of the proposal policy (that collected those training trajectories) given noisy image $x_t$, which 0814 abbreviates to $v_\text{base}$.

Substituting Eq. 28 and Eq. 29 into Eq. 24 gives:

$$
\nabla V_t = - \frac{1-t}{t} \mathbb{E}_Q [A(x_0)(v-v_\text{base}) \mid x_t]
\tag{30}
$$

Thus a one sample estimator, named as the deterministic estimator by 0814 as it contains no local reverse-SDE Brownian-motion terms($\mathrm{d}W_t / \sqrt{\Delta t}$), looks like this

$$
\widehat{\nabla V}^\text{det} = - \frac{1-t}{t} A(x_0)(v-v_\text{old})
\tag{31}
$$

where $v_\text{base}$ is replaced by $v_\text{old}$ because we assume the diffusion model, which is exactly trained to fit $v_\text{base}$ is sufficiently capable to represent the conditional expected velocity field associated with its own generations (namely $v_\text{base}$). (Yet more recent work like [DiffusionART](https://arxiv.org/abs/2610.12572) discussed whether we can take this substitution for granted.)

**To sum up, 0814 identified two different value gradient estimators.** The stochastic estimator identifies which Brownian kick in the sampling process created the good image, while the deterministic estimator pulls the model velocity directly to align with those that generated great endpoints.

$$
\boxed{\widehat{\nabla V}^\text{sto} = \frac{A(x_0) \epsilon_r}{g_r \sqrt{\Delta r}}}.
$$

$$
\boxed{
  \widehat{\nabla V}^\text{det} = - \frac{1-t}{t} A(x_0)(v-v_\text{old})
}
$$

### 1.5 Estimators to algorithms: Flow-GRPO and AWM

We have so far deliberately stayed at the level of an **infinitesimal policy perturbation**. From Eq. 19, of the drift is parameterized by $\theta$,

$$
\delta\mu_r = \nabla_\theta\mu_\theta(y_r,r)\,\delta\theta,
$$

then

$$
\nabla_\theta J_{\mathrm{policy}} =
\mathbb E_Q
[\int_0^1 (\nabla_\theta\mu_\theta)^\top \nabla V_r\,dr].
\tag{32}
$$

So estimating $\nabla V$ is translated from the geometry (SDE drift) to practical parameter updates. This gives us a clean way to read existing diffusion-RL algorithms. Under the first-order approximation, the question is essentially: **Which estimator of $\nabla V$ does the algorithm implicitly use?**

(For simplicity I will ignore the finite-step effects of clipping and KL regularization for the time being)

#### [Flow-GRPO](https://arxiv.org/abs/2505.05470): stochastic estimator

Flow-GRPO starts from the most direct RL interpretation of the denoising trajectory. A state is $(c,t,x_t)$, an action is the next denoised state $x_{t-\Delta t}$.(Introducing the $y_r = x_t, r=1-t$ notation once more.) Ignoring clipping for the moment, its policy ratio is

$$
\rho_r(\theta)
=
\frac{
p_\theta(y_{r+\Delta r}\mid y_r)
}{
p_{\mathrm{old}}(y_{r+\Delta r}\mid y_r)
}.
\tag{33}
$$

This is exactly the Gaussian transition we already used in deriving Girsanov (Eq. 4). At the rollout policy $\theta=\theta_{\mathrm{old}}$,

$$
y_{r+\Delta r}
=
y_r+\mu_{\mathrm{old}}\Delta r
+
g_r\sqrt{\Delta r}\,\epsilon_r.
$$

Taking the score of this Gaussian gives

$$
\begin{aligned}
\nabla_\theta
\log
p_\theta(y_{r+\Delta r}\mid y_r)
\big|_{\theta=\theta_{\mathrm{old}}}
&=
(\nabla_\theta\mu_{\mathrm{old}})^\top
\frac{
y_{r+\Delta r}-y_r-\mu_{\mathrm{old}}\Delta r
}{
g_r^2
}
\\
&=
(\nabla_\theta\mu_{\mathrm{old}})^\top
\frac{\sqrt{\Delta r}}{g_r}\epsilon_r.
\end{aligned}
\tag{34}
$$

Multiplying by the terminal advantage $A$,

$$
A(y_1)
\nabla_\theta\log p_{\mathrm{old}} =
\Delta r
(\nabla_\theta\mu_{\mathrm{old}})^\top
\boxed{
\frac{A(y_1)\epsilon_r}
{g_r\sqrt{\Delta r}}}
\tag{35}
$$

And we recognize the boxed term as the stochastic value gradient estimator from Eq. 22.

**So, to first order around the rollout policy, Flow-GRPO can be read as a discrete implementation of the stochastic value-gradient estimator.**

And once more, using the stochastic estimator creates a high-variance issue. Flow-GRPO asks every random Brownian kick along a successful trajectory:

*Were you one of the reasons this image eventually became good?*

Only the tiny reward-correlated component survives in expectation; the unrelated components cancel after averaging many trajectories.

> The actual Flow-GRPO algorithm of course contains more than Eq. 35: it uses clipped ratios, a reference-model KL penalty, denoising reduction etc. Yet they are not needed to understand which of our two first-order value-gradient estimators it is using. In addition [Dance-GRPO](https://arxiv.org/abs/2505.07818) also shares substantial resemblance with Flow-GRPO and can be explained in the same way. For more variations of Diffusion GRPOs please refer to the [Diffusion GRPO survey](https://arxiv.org/abs/2603.06623).

#### [AWM](https://arxiv.org/abs/2509.25050): Endpoint as action

AWM is still GRPO in some sense, but instead of regarding every $x_{t-\Delta t}$ as an RL action, it treats the entire generated clean image as the sequence-level action:
$x_0\sim\pi_{\mathrm{old}}(x_0\mid c),$
and the sequence-level GRPO objective is written as

$$
J_{\mathrm{GRPO}}
=
\mathbb E
[
\frac{\pi_\theta(x_0\mid c)}
{\pi_{\mathrm{old}}(x_0\mid c)}
A(x_0,c)
]
-
\beta D_{\mathrm{KL}}
(\pi_\theta\|\pi_{\mathrm{ref}}).
\tag{36}
$$

(and we still ignore that KL regularizer for the time being.) The problem, of course, is exactly where this blog began: $\pi_\theta(x_0\mid c)$ is not tractable, and AWM therefore replaces its log likelihood with **an ELBO surrogate**,

$$
\log\hat\pi_\theta(x_0\mid c)
=
-\mathbb E_{t,\epsilon}
[w(t)\|v_\theta(x_t,t,c)-(\epsilon-x_0)\|^2]
+\text{const}.
\tag{37}
$$

The theoretical ELBO weight for a rectified flow is $w_{\mathrm{ELBO}}(t)=\frac{1-t}{t},$ although AWM claims that simpler alternatives such as $1$ or $t$ can work better empirically.

(Ignoring the KL term), the first variation of the loss is

$$
\delta J_{\mathrm{AWM}}
= A(x_0) \delta \log \hat\pi_\theta(x_0)
\tag{38}
$$

At $\theta=\theta_{\mathrm{old}}$ (the $\epsilon-x_0$ in Eq. 37 is simply $v$),

$$
\begin{align}
\delta \log \hat\pi_\theta(x_0)
&= -\delta \mathbb E_t
[w(t)\|v_\theta(x_t,t,c)-(\epsilon-x_0)\|^2] \\
&= -2 \mathbb E_t [w(t)(v_\mathrm{old} - v)^\top \delta v_\theta]
\end{align}
\tag{39}
$$

Therefore we obtain from Eq. 38 and 39 the first variation of the loss:

$$
\delta J_{\mathrm{AWM}}
=
2\mathbb E
\left[
A(x_0)\,w(t)
(v-v_{\mathrm{old}})^\top
\delta v_\theta
\right].
\tag{40}
$$

And in our deterministic value-gradient estimator (Eq. 31),

$$
\widehat{\nabla V}^{\mathrm{det}}
=
-\frac{1-t}{t}
A(x_0)(v-v_{\mathrm{old}}).
$$

**the same reward-times-velocity residual appears**: $A(x_0)(v-v_{\mathrm{old}}).$

> In fact, if we consider *how* $\mu$ is given from $v$ using the flow-matching SDE,with the standard $\eta_t=1$ used in the 0814 derivation, switching from forward time $t$ to sampling time $r=1-t$ gives the reverse-drift perturbation (a more detailed explanation later in Section 2)
>
> $$
> \delta\mu_r=-(1+\eta_t)\delta v_\theta = -2\delta v_\theta. \tag{41}
> $$
>
> Substituting Eq. 31 into Eq. 19 therefore gives
>
> $$
> \begin{aligned}\delta J_{\mathrm{policy}}
> &=\mathbb E\int[-\frac{1-t}{t}A(v-v_{\mathrm{old}})]^\top(-2\delta v_\theta)\,dt\\
> &=2\mathbb E\int\frac{1-t}{t}A(v-v_{\mathrm{old}})^\top\delta v_\theta\,dt.
> \end{aligned}\tag{42}
> $$
>
> If the ELBO weight satisfies $w(t)=(1-t)/t$, this is exactly the local variation in Eq. 40. In spirit, we can say AWM is an instance of the deterministic value gradient estimator. (Note a small discrepancy here with the empirical claim of $w(t)=1$ works well: we will explain it later in Section 2.)

---

We can therefore summarize the first-order story as

$$
\begin{array}{ccc}
\text{Flow-GRPO}
&
\longrightarrow
&
\displaystyle
\widehat{\nabla V}^{\mathrm{sto}}
=
\frac{A\epsilon_r}
{g_r\sqrt{\Delta r}}
\\[12pt]
\text{AWM}
&
\longrightarrow
&
\displaystyle
\widehat{\nabla V}^{\mathrm{det}}
=
-\frac{1-t}{t}
A(v-v_{\mathrm{old}})
\end{array}
$$

The first asks which **random reverse-process perturbation** (therefore written with $r$) happened to correlate with terminal reward. The second uses diffusion's known forward corruption kernel (therefore written with $t$) to ask which **endpoint-specific velocity residual** is associated with terminal reward.

They might look like very different algorithms because they target at different random variables during training. Under the first-order path-space analysis, however, both happen to be ways of estimating the same object $\nabla V$.

## 2. From local policy improvement to finite-update objectives

> Section 2 will take a slightly different route from the mathematical explanation in 0814, as observing a common structure (subsections 2.1 to 2.3) likely comes first, and then the mathematical justification (centered around the Girsanov theorem, subsections 2.4 to 2.5) follows.

### 2.1 First order isn't enough: what AWM and NFT actually optimize

Section 1 focused on the question of *where to move* around the proposal policy, which is a purely first-order (infinitesimal) question. However, practical RL algorithms are rarely purely on-policy, i.e. they do not take a single infinitesimal step and resample, but instead minimize a surrogate loss over multiple finite updates. Once we take finite steps away from the proposal $v_{\mathrm{old}}$, the **quadratic part** of the loss starts to matter.

In fact, we already stepped over a quadratic term in Section 1 without paying much attention to it. In Eqs. 39–40, we differentiated AWM's squared matching objective at $v_\theta = v_{\mathrm{old}}$. There a quadratic term slips away in that first-order calculation.

To see the algebra clearly, let's give short names to the three quantities from the end of Section 1:
* $\Delta v_\theta := v_\theta - v_{\mathrm{old}}$ is our finite policy displacement from the rollout model,
* $d := v - v_{\mathrm{old}}$ is the target velocity residual from Eq. 31, and
* $s(t) := \frac{1-t}{t}$ is the theoretical ELBO time weight.

Expanding AWM's signed matching loss around $v_{\mathrm{old}}$ gives:

<a id="eq-awm-expansion"></a>

$$
\begin{aligned}
\mathcal S_{\mathrm{AWM}}
&=\mathbb E_{x_0,c,t,\epsilon}\left[A w(t)\|v_\theta-v\|^2\right]\\
&=\mathbb E\left[
-2A w(t)d^\top\Delta v_\theta
+A w(t)\|\Delta v_\theta\|^2
\right]+C,
\end{aligned}
\tag{43}
$$

where $C$ are terms independent of $\theta$. The first term is the linear, reward-correlated term we already studied in Section 1. The second term is quadratic in $\Delta v_\theta$, which at $\Delta v_\theta = 0$ has zero gradient. Notice that its coefficient $A w(t)$ is signed. Whenever a trajectory has a negative advantage ($A < 0$), the quadratic term turns concave, actively rewarding the network for moving $\Delta v_\theta$ infinitely far away from the target.

Now compare this with [**DiffusionNFT**](https://arxiv.org/abs/2509.16117). It also looks like velocity regression, but it constructs its loss via two implicit policies and the behavior away from $v_{\mathrm{old}}$ is completely different. Let $\rho \in [0,1]$ denote NFT's normalized reward (written as $r$ in the original paper, renamed here to avoid confusion with sampling time $r=1-t$), define the centered advantage $A_{\mathrm{NFT}} := 2\rho - 1 \in [-1, 1]$, and let $\beta > 0$ be NFT's mixing parameter:

<a id="eq-nft-objective"></a>

$$
\begin{aligned}
\mathcal S_{\mathrm{NFT}}(\theta)
&=\mathbb E\left[
\rho\|v_\theta^+-v\|^2+(1-\rho)\|v_\theta^--v\|^2
\right],\\
v_\theta^+&=(1-\beta)v_{\mathrm{old}}+\beta v_\theta
=v_{\mathrm{old}}+\beta\Delta v_\theta,\\
v_\theta^-&=(1+\beta)v_{\mathrm{old}}-\beta v_\theta
=v_{\mathrm{old}}-\beta\Delta v_\theta.
\end{aligned}
\tag{44}
$$

Expanding a single training example in terms of $\Delta v_\theta$ and $d$ gives

<a id="eq-nft-expansion"></a>

$$
\begin{aligned}
l_{\mathrm{NFT}}
&=\rho\|\beta\Delta v_\theta-d\|^2
+(1-\rho)\|-\beta\Delta v_\theta-d\|^2\\
&=-2\beta A_{\mathrm{NFT}}d^\top\Delta v_\theta
+\beta^2\|\Delta v_\theta\|^2+\|d\|^2.
\end{aligned}
\tag{45}
$$

Because $\rho + (1-\rho) = 1$, the two branches contribute the **exact same positive quadratic coefficient** $\beta^2$, while their cross terms subtract to give $\rho - (1-\rho) = A_{\mathrm{NFT}}$. Completing the square we get

<a id="eq-nft-target"></a>

$$
l_{\mathrm{NFT}}
=\beta^2\left\|
\Delta v_\theta-\frac{A_{\mathrm{NFT}}}{\beta}d
\right\|^2
+(1-A_{\mathrm{NFT}}^2)\|d\|^2.
\tag{46}
$$

And now we can compare Eq. 46 (NFT) and Eq. 43 (AWM). For NFT, whether the advantage is positive or negative, the loss is always convex with constant positive curvature $2\beta^2 I$, pulling the displacement $\Delta v_\theta$ toward a finite target $\frac{A_{\mathrm{NFT}}}{\beta}d$ (toward $v$ when $A_{\mathrm{NFT}} > 0$, and away when $A_{\mathrm{NFT}} < 0$).

| Matching surrogate | Linear term (1st order) | Quadratic coefficient (2nd order) |
| --- | --- | --- |
| **AWM** | $-2A w(t)d^\top\Delta v_\theta$ | $A w(t)$ *(signed: unstable for $A<0$)* |
| **DiffusionNFT** | $-2\beta A_{\mathrm{NFT}}d^\top\Delta v_\theta$ | $\beta^2 > 0$ *(always positive)* |


### 2.2 The observed linear-quadratic design space

AWM (Eq. 43) and NFT (Eq. 45) now have a same linear-quadratic form in $\Delta v_\theta$. The linear term that aligns with our value-gradient estimator $\widehat{\nabla V}_t$, plus a quadratic term $\|\Delta v_\theta\|^2$.

Organizing existing methods into this shared template is, in my view, the central practical contribution of 0814. Written as a continuous-time loss to minimize, the unified surrogate is:

<a id="eq-unified-surrogate"></a>

$$
\boxed{
\mathcal S_{\mathrm{unified}}(\theta)
=\mathbb E\int_0^1
\left[
\hat w_2(t)\widehat{\nabla V}_t^\top\Delta v_\theta
+\hat w_1(A,t)\|\Delta v_\theta\|^2
\right]\,\mathrm d t.
}
\tag{47}
$$

*(Here $\hat w_2$ can also depend on the sampler schedule $\eta_t$, which is suppressed in the notation.)*

AWM and NFT drop straight into Eq. 47 by simple algebra. Recall from Eq. 31 that the deterministic estimator is $\widehat{\nabla V}^{\mathrm{det}} = -s(t)Ad$. For NFT, dividing Eq. 45 by the positive constant $\beta^2$ gives:

<a id="eq-nft-framework"></a>

$$
\frac{\mathcal S_{\mathrm{NFT}}}{\beta^2}
=\mathbb E\left[
\|\Delta v_\theta\|^2
+\frac{2}{\beta s(t)}
\left(\widehat{\nabla V}^{\mathrm{det}}_{\mathrm{NFT}}\right)^\top
\Delta v_\theta
\right]+C,
\qquad
\widehat{\nabla V}^{\mathrm{det}}_{\mathrm{NFT}}
:=-s(t)A_{\mathrm{NFT}}d.
\tag{48}
$$

Thus, every algorithm in this family can be specified by five coordinates: **$(\eta_t, v_{\mathrm{base}}, \widehat{\nabla V}_t, \hat w_1, \hat w_2)$**. The table in 0814 maps existing methods into this exact coordinate system:

![table of methods]({{ '/assets/diffusionrl/methodtable.png' | relative_url }})

This gives us a clean taxonomy and design space:
- **$\eta_t$ (the sampler):** Which reverse process supplies the training states (SDE vs. ODE)?
- **$v_{\mathrm{base}}$ (the anchor):** What defines zero displacement ($\Delta v_\theta = v_\theta - v_{\mathrm{base}}$)? For path-ratio methods, it is the rollout policy $v_{\mathrm{old}}$. For the exact forward-process identity (Eq. 29), it is the true conditional mean $\mathbb E_Q[v\mid x_t,c]$ (replacing that mean with $v_{\mathrm{old}}$ is an approximation that relies on the model already fitting its own generations well).
- **$\widehat{\nabla V}_t$ (the improvement direction):** Do we use the high-variance stochastic estimator $\widehat{\nabla V}^{\mathrm{sto}}$ or the lower-variance deterministic estimator $\widehat{\nabla V}^{\mathrm{det}}$?
- **$\hat w_1(A,t)$ (the quadratic weight):** What is the sign and strength of the quadratic displacement term $\|\Delta v_\theta\|^2$?
- **$\hat w_2(t)$ (the linear weight):** How strongly do we push along the value-gradient estimate $\widehat{\nabla V}_t$ at timestep $t$?

> **A quick bookkeeping note on $\Delta t$ in the table:** Because Eq. 47 is written as a continuous integral $\int_0^1 [\dots]\,\mathrm d t$, the weights $\hat w_1$ and $\hat w_2$ are *densities (to time)*. When an implementation samples discrete timesteps of width $\Delta t$, a single width contributes $\left[\hat w_2(t)\widehat{\nabla V}_t^\top\Delta v_\theta+\hat w_1(A,t)\|\Delta v_\theta\|^2\right]\Delta t$. Therefore, to match the per-step discrete losses in Eq. 43 (AWM) and Eq. 48 (NFT), 0814's table divides the discrete coefficients by $\Delta t$:
>
> <a id="eq-bin-coefficients"></a>
>
> $$
> \begin{array}{c|cc}
> & \hat w_1 & \hat w_2\\
> \hline
> \mathrm{AWM} & A w(t)/\Delta t & 2w(t)/[s(t)\Delta t]\\
> \mathrm{NFT}/\beta^2 & 1/\Delta t & 2/[\beta s(t)\Delta t]
> \end{array}
> \tag{49}
> $$
>
> Setting $w(t)=1$ in the AWM row immediately reproduces the table entries $A/\Delta t$ and $2/[s(t)\Delta t]$.

### 2.3 Weighting is part of the algorithm (and why $w(t)=1$ works)

Of course not all choices of $(\hat w_1, \hat w_2)$ behave equally well in practice:

1. **The sign of $\hat w_1$ governs stability:** Because NFT uses a positive constant $\hat w_1 = 1/\Delta t$, its quadratic term always acts as a genuine trust-region penalty anchoring $v_\theta$ near $v_{\mathrm{old}}$. AWM's signed weight $\hat w_1 = A w(t)/\Delta t$ flips sign for $A < 0$, which explains why AWM needs an explicit external KL regularizer to avoid collapsing, and why unconstrained AWM is notoriously [unstable during training](https://github.com/X-GenGroup/Flow-Factory/blob/main/guidance/algorithms.md#training-stability).
   
2. **The product $\hat w_2(t)\widehat{\nabla V}_t$ governs gradient variance across $t$:** Even when an estimator is unbiased in expectation, practical updates use noisy single-sample Monte Carlo estimates. And this will try to answer why AWM found empirically that a flat weight $w(t)=1$ works much better than the theoretical ELBO weight $w(t) = s(t) = \frac{1-t}{t}$ (Section 1.5).

Let's consider one parameter update during training. Let $K_\theta(x_t,t) := \frac{\partial v_\theta(x_t,t)}{\partial\theta}$ be the network Jacobian. A single timestep bin of width $\Delta t$ contributes the parameter gradient:

<a id="eq-parameter-gradient"></a>

$$
\Delta t\,K_\theta^\top
\left[
\hat w_2(t)\widehat{\nabla V}_t
+2\hat w_1(A,t)\Delta v_\theta
\right].
\tag{50}
$$

If we use AWM with the theoretical ELBO weight $w(t) = s(t) = \frac{1-t}{t}$, then $\hat w_2 = \frac{2}{\Delta t}$ looks harmlessly constant, but the actual per-example loss integrand (up to $1/\Delta t$) is:

<a id="eq-elbo-weighting"></a>

$$
s(t)A\left[
\|\Delta v_\theta\|^2-2d^\top\Delta v_\theta
\right]
= \frac{1-t}{t}A\left[
\|\Delta v_\theta\|^2-2d^\top\Delta v_\theta
\right].
\tag{51}
$$

Because $s(t) = \frac{1-t}{t}$ blows up as $t \to 0$, both the linear update and the quadratic term inherit a $1/t$ near the clean image endpoint ($t=0$), massively amplifying Monte Carlo noise at small $t$. 

By contrast, when we set $w(t) = 1$ in AWM (or use NFT), the weight $\hat w_2 = \frac{2}{s(t)\Delta t}$ looks singular on paper only because it is written to **cancel out** the $-s(t)$ factor sitting inside $\widehat{\nabla V}^{\mathrm{det}} = -s(t)Ad$! Their product is simply $\Delta t\,\hat w_2(t)\widehat{\nabla V}^{\mathrm{det}} = -2Ad$, which is completely bounded across all $t \in [0,1]$. This is the essence of 0814's scale-bounded weighting principle: the combined effective coefficient multiplying the noisy residual $d$ and displacement $\Delta v_\theta$, must be bounded.

### 2.4 Quadratic term, from theory

Up to this point, everything in Section 2 has been pure algebra on existing losses. Now let's ask the theoretical question we postponed from Section 1.2: Why did those quadratic terms appear in the first place?

To answer that, recall the Girsanov path-space log-ratio (Eq. 11) and write it in velocity coordinates. For a flow-matching SDE in sampling time $r = 1-t$, the model velocity $v_\theta(x_t,t)$ enters the reverse drift linearly:

$$
\begin{aligned}
\mu_\theta(y_r,r)&=c_r(y_r)-(1+\eta_t)v_\theta(x_t,t),\\
g_r^2&=\frac{2t\eta_t}{1-t},\qquad t=1-r,\\
\mathrm d y_r&=\mu_\theta(y_r,r)\,\mathrm d r+g_r\,\mathrm d W_r.
\end{aligned}
\tag{52}
$$

 where $c_r(y_r)$ is independent of the velocity. Since $c_r(y_r)$ is shared by $P_\theta$ and the proposal $Q$ (generated by $v_{\mathrm{base}}$), the drift difference is simply proportional to the velocity displacement:

$$
\Delta\mu_r
=\mu_\theta-\mu_{\mathrm{base}}
=-(1+\eta_t)(v_\theta-v_{\mathrm{base}})
=-(1+\eta_t)\Delta v_\theta.
\tag{53}
$$

(For $v_{\mathrm{base}} = v_{\mathrm{old}}$ and $\eta_t = 1$, this gives $\Delta\mu_r = -2\Delta v_\theta$, the exact conversion we used in Eq. 41.)

Now let's compute the normalized drift change $\frac{\Delta\mu_r}{g_r}$ that appears inside Girsanov's theorem (Eq. 11):

$$
\frac{\Delta\mu_r}{g_r}
=-\frac{1+\eta_t}{\sqrt{2t\eta_t/(1-t)}}\Delta v_\theta
=-\sqrt{\frac{(1+\eta_t)^2(1-t)}{4t\eta_t}}\,\Delta v_\theta.
$$

Collecting that scalar prefactor into $\widetilde w_t$ we get

$$
\widetilde w_t:=\frac{(1+\eta_t)^2(1-t)}{4t\eta_t},
\qquad
\frac{\Delta\mu_r}{g_r}=-\sqrt{2\widetilde w_t}\,\Delta v_\theta,
\qquad
\frac12\left\|\frac{\Delta\mu_r}{g_r}\right\|^2
=\widetilde w_t\|\Delta v_\theta\|^2.
\tag{54}
$$

> (Notice that for standard SDE sampling $\eta_t = 1$, $\widetilde w_t$ is once again $\frac{1-t}{t} = s(t)$! Intuitively, because the sampling noise $g_r$ shrinks to zero as $t \to 0$, any fixed velocity change $\Delta v_\theta$ becomes infinitely distinguishable against the vanishing noise near the clean endpoint.)

Substituting Eq. 54 into Eq. 11 gives the exact log-likelihood ratio $M_1(\theta) := \log\frac{\mathrm dP_\theta}{\mathrm dQ}$ in velocity coordinates:

$$
\begin{aligned}
M_1(\theta)
:=\log\frac{\mathrm dP_\theta}{\mathrm dQ}
={}&-\int_0^1\sqrt{2\widetilde w_{1-r}}\,
\Delta v_\theta(x_{1-r},1-r)^\top\mathrm dW_r\\
&-\int_0^1\widetilde w_{1-r}
\|\Delta v_\theta(x_{1-r},1-r)\|^2\,\mathrm d r.
\end{aligned}
\tag{55}
$$

$M_1(\theta)$ consists of two terms: the first is the stochastic Brownian alignment term, and the second is the quadratic path-energy cost of moving $P_\theta$ away from $Q$.

Recall the true policy objective $\mathbb E_Q\left[e^{M_1(\theta)}A\right]$ and now we have $M_1$.
On my first read of 0814, prompted by phrasing like *"explicitly ... exactly obtained"*, I thought we simply linearized the exponential via $e^{M_1} \approx 1 + M_1$ inside the expectation and flipped the sign to minimize a surrogate loss:

$$
\begin{aligned}
\mathcal S_{\log}(\theta)
:=-\mathbb E_Q[A M_1(\theta)]
=\mathbb E_Q\Bigg[
&A\int_0^1\sqrt{2\widetilde w_{1-r}}\,
\Delta v_\theta^\top\mathrm dW_r\\
&+A\int_0^1\widetilde w_{1-r}
\|\Delta v_\theta\|^2\,\mathrm d r
\Bigg].
\end{aligned}
\tag{56}
$$

Though to be fair to 0814, Proposition 1 instead uses a trust-region condition $\lvert \exp(M_1(\theta^-)) - 1 \rvert < \epsilon$ to argue that the *gradients* of $-\mathbb E_Q[e^{M_1}A]$ and $\mathcal S_{\log}(\theta)$ match in the limit $\epsilon \to 0$, and then applies our martingale identity (Eq. 18) to rewrite the Brownian integral in terms of $\nabla V_t$:

$$
\mathcal S_{\log}(\theta)
=\mathbb E_Q\int_0^1
\left[
(1+\eta_t)\nabla V_t^\top\Delta v_\theta
+A\widetilde w_t\|\Delta v_\theta\|^2
\right]\,\mathrm d t.
\tag{57}
$$

Sure enough, Eq. 57 is an instance of the unified template (Eq. 47) with $\hat w_2 = 1 + \eta_t$ and $\hat w_1 = A\widetilde w_t$ (which at $\eta_t=1$ reproduces AWM's ELBO-weighted loss in Eq. 51). 

But here is a discrepancy. Whether you arrive at Eq. 57 via the naive $e^{M_1} \approx 1 + M_1$ shortcut or via 0814's Proposition 1, the proof only establishes **first-order gradient agreement** at the proposal ($\Delta v_\theta \to 0$). Yet the paper's prose states an exact equality and calls the quadratic term $A\widetilde w_t\|\Delta v_\theta\|^2$ an "off-policy regulariser" that "anchors" $v_\theta$ to $v_{\mathrm{base}}$. 

That is a second-order claim about how the loss behaves away from the proposal. And as we already saw, once the nonnegative Girsanov energy $\widetilde w_t\|\Delta v_\theta\|^2$ is multiplied by a signed advantage $A$, it isn't even a positive penalty. Why did a first-principles Girsanov derivation produce a broken second-order regularizer?

### 2.5 The Girsanov derivation breaks at second order

The derivation breaks because keeping the quadratic term inside $M_1$ while dropping the quadratic term of the exponential $e^{M_1} = 1 + M_1 + \frac{1}{2}M_1^2 + \dots$ is an inconsistent Taylor expansion. It carries a second-order term ($\|\Delta v_\theta\|^2$) into a first-order approximation but meanwhile leaves behind another second-order term of the exact same magnitude.

For convenience in counting the orders (and not writing constants all over), let's perturb the drift along a fixed adapted direction $h_r$ by a scalar step size $\lambda$ (so $\mu_\lambda = \mu_Q + \lambda h$):

$$
Z:=\int_0^1\frac{h_r^\top}{g_r}\,\mathrm dW_r,
\qquad
C_h:=\int_0^1\frac{\|h_r\|^2}{g_r^2}\,\mathrm d r,
\qquad
M_\lambda=\lambda Z-\frac{\lambda^2}{2}C_h.
\tag{58}
$$

Here $\lambda Z$ is our $O(\lambda)$ stochastic integral and $\frac{\lambda^2}{2}C_h$ is our $O(\lambda^2)$ Girsanov quadratic energy. Now we can see the difference between the true second-order expansion and the log-surrogate $1 + M_\lambda$:

$$
e^{M_\lambda}
=1+\lambda Z+\frac{\lambda^2}{2}(Z^2-C_h)+o(\lambda^2),
\qquad
1+M_\lambda
=1+\lambda Z-\frac{\lambda^2}{2}C_h.
\tag{59}
$$

Multiplying by the terminal advantage $A$ and taking expectations under $Q$, the true policy objective $J(\lambda) := \mathbb E_Q[A e^{M_\lambda}]$ and the log-surrogate $\widetilde J(\lambda) := \mathbb E_Q[A(1+M_\lambda)]$ expand as:

$$
\begin{aligned}
J(\lambda)
&=J(0)+\lambda\mathbb E_Q[AZ]
+\frac{\lambda^2}{2}\mathbb E_Q[A(Z^2-C_h)]
+o(\lambda^2),\\
\widetilde J(\lambda)
&=J(0)+\lambda\mathbb E_Q[AZ]
-\frac{\lambda^2}{2}\mathbb E_Q[AC_h].
\end{aligned}
\tag{60}
$$

Look at the $O(\lambda^2)$ terms in Eq. 60. The first-order gradients $\mathbb E_Q[AZ]$ at $\lambda = 0$ match by construction, but the second-order terms differ by $\frac{\lambda^2}{2}\mathbb E_Q[AZ^2]$:

$$
\widetilde J(\lambda)-J(\lambda)
=-\frac{\lambda^2}{2}\mathbb E_Q[AZ^2]+o(\lambda^2).
$$

> An explanation for Eq. 60. Though Ito isometry guarantees that $\mathbb E_Q[Z^2] = \mathbb E_Q[C_h]$, multiplying the difference with the advantage (which is usually correlated with the path that led to the clean image)
> $$
>  \mathbb E_Q[Z^2-C_h]=0,
>  \qquad\text{but generally}\qquad
>  \mathbb E_Q[A(Z^2-C_h)]\ne0.
>  \tag{61}
>  $$

We can see this difference between $\widetilde J(\lambda)$ and $J(\lambda)$ in two tiny toy examples:

+ Centered quadratic reward ($A$ has zero mean, yet the surrogate misses the curvature completely): Let $\mathrm dY_r=\lambda\,\mathrm d r+\mathrm dW_r$ with $Y_0=0$, and take the centered advantage $A(y)=y^2-1$. Under the proposal ($\lambda=0$), $Z=W_1$, $C_h=1$, and $\mathbb E_Q[A]=0$. Yet

  $$
  J(\lambda)=\mathbb E[(W_1+\lambda)^2-1]=\lambda^2,
  \qquad
  \mathbb E_Q[A(Z^2-C_h)]
  =\mathbb E[(W_1^2-1)^2]=2.
  \tag{62}
  $$

  Here the true objective $J(\lambda) = \lambda^2$ is strictly convex, driven entirely by the omitted $AZ^2$ correlation. Meanwhile, in the log-surrogate $\widetilde J(\lambda)$, both $\mathbb E_Q[AZ] = 0$ and $\mathbb E_Q[AC_h] = \mathbb E_Q[A] = 0$ vanish, so the surrogate is completely flat.

+ Constant reward shift ($J$ is flat, yet the surrogate gives signed curvature): Conversely, suppose $A \equiv a \ne 0$ is a constant baseline. Then the true expected reward is independent of $\theta$, while the log-surrogate becomes proportional to the KL divergence:

  $$
  J(P_\theta)=a,
  \qquad
  \mathcal S_{\log}(\theta)
  =-a\mathbb E_Q[M_1(\theta)]
  =aD_{\mathrm{KL}}(Q\|P_\theta).
  \tag{63}
  $$

  In the true objective, the omitted $Z^2$ term cancels $C_h$ so that shifting the reward baseline has zero effect. In the log-surrogate, dropping $Z^2$ leaves behind a fake curvature $a D_{\mathrm{KL}}(Q\|P_\theta)$ whose sign flips with the sign of $a$!

### Summary of Section 1 & 2

Stepping back, this blog discusses 0814 and its unified table:
1. **The first-order Girsanov + martingale story (Section 1).** It explains why both trajectory-level REINFORCE ($\widehat{\nabla V}^{\mathrm{sto}}$) and endpoint velocity matching ($\widehat{\nabla V}^{\mathrm{det}}$) estimate the exact same policy improvement direction $\nabla V_t$ around the proposal.
2. **The second-order Girsanov story (Section 2.4).** Keeping the $O(\lambda^2)$ Girsanov energy $A C_h$ while dropping the $O(\lambda^2)$ outer-product term $A Z^2$ does not give the true curvature of the RL objective, nor does it yield a valid trust-region anchor (since $A C_h$ flips sign when $A < 0$).
3. **And that makes 0814's empirical design space ($\mathcal S_{\mathrm{unified}}$, Eq. 47) more defensible.** Once we recognize that first-principles path-space theory only pins down the linear improvement direction and does not dictate the quadratic coefficient, so it leaves room for us to choose the quadratic coefficient. And of course NFT's choice seems much easier to justify than AWM's signed one. (And that is quite a satisfactory read.😽)


## 3. Beyond PPO-style
There are far more RL algorithms (usually older ones) that this framework does not discuss/cover.

Interestingly reviewers of NFT on OpenReview also noted a similarity between KTO and NFT, and the authors responded that KTO could be seen as a special instance.

### 3.1 Diffusion DPO
(TBD)

(Here's the thing: DPO actually also uses the ELBO surrogate as in AWM, this actually makes me wonder *what kind of a role* ELBO plays in this unified framework and possibly how using the same surrogate makes two methods closer than initially imagined)

### 3.2 Diffusion KTO
(TBD)

(KTO is NFT wrapped in a Utility function. It's actually worth exploring HOW that utility function might be desirable or not, an obvious drawback would be that it ruined that clean finite target regression)

### 3.3 Others? Not sure

(TBD)

(CaPO, which claims to inherit IPO)

(DPPO?)

(I have procrastinated this so long that DiffusionART came out)

### 3.4 Empirical analysis

I am actually thinking of a comparison between DPO KTO (CaPO/IPO) NFT and possibly more, an analogy to the huggingface blog in early 2024 that compared DPO KTO and IPO.


(Connection with RAM)

> Ends here.

<!-- {% endcapture %}{% include math-markdown.html content=math_content %} -->
