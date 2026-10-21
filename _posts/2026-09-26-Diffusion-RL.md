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

(09/26/26: Found one newer survey, updated Nov. 2025. [XueZeyue/Awesome-Visual-Generation-Alignment-Survey](https://github.com/XueZeyue/Awesome-Visual-Generation-Alignment-Survey).)

(10/18/26: The topic of this survey is probably what I am looking for, but it is, from my point of view, not very insightful. [Diffusion RL Survey Oct. 15th](www.arxiv.org/abs/2610.21539))

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


where $\tau$ denotes the denoising trajectory. That observation alone already explains half of the Diffusion RL problem (peeled off the maths), I daresay.

Now we can handle the RL problem with that product. We have trajectories collected from an older policy $Q$, but again, what we need to calculate $\mathbb{E}_{x_0 \sim P_\theta(x_0)}[R(x_0)]$ or in path-space view $\mathbb{E}_{\tau \sim P_\theta(\tau)}[A(x_0(\tau))]$, something sampled under $P_\theta$ not $Q$. Luckily, it's easy to derive:

$$
\mathbb{E}_{P_\theta(\tau)} A(x_0(\tau)) = \mathbb{E}_{Q(\tau)}{[\frac{P_\theta(\tau)}{Q(\tau)}A(x_0(\tau))]} = \mathbb{E}_{Q(\tau)} [\prod_t \frac{p_\theta (x_{t-\Delta t}  \mid  x_t)}{q(x_{t-\Delta t}  \mid  x_t)} A(x_0(\tau))] \tag{4}
$$

(The initial noise distribution $p_1(x_1)$ cancels out in $P_\theta$ and $Q$. **This is simply importance sampling and the direct source of Flow-GRPO's per step optimization.** But let's dive deeper with 0814 before returning to any specific method.)

And now it's time to replace the abstract $p$'s and $q$'s with continuous diffusion SDEs. Note that as $Q$ is an older version of the model, the stochasticity schedule stays fixed throughout the training as a selected implementation design before any training occurs at all. Thus we can assert that the diffusion coefficient $g$'s are the same for $P$ and $Q$.

**For convenience in discussing Ito calculus and avoiding $\mathrm{d}t<0$ problems, I follow the conventions of Appendix C of 0814, where $r := 1-t$ and $y_r = x_{1-t}$)** Recall the target form; under this $r$ convention it would be:

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

Expanding the R-N derivative $\frac{\mathrm{d}P}{\mathrm{d}Q}$ and **keeping only the first order $\Delta \mu$ pertubation** (*and we will return to the second time in section 2*) yields (for notation convenience I will abbreviate $\mathbb{E}_{P_\theta(\tau)} A(x_0(\tau))$ to $J_{policy}$):

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

**We have proved that rewarding all the random Brownian movements (though noisy) along a good trajectory is, in expectation and first order to $\Delta \mu$, the same to moving the drift directly in the direction that increases expected terminal advantage.** The central amount we care about is now $\nabla V_r$.

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

And I think the following part (represented by the idea of AWM and NFT) is the real genius. We have been asking which Brownian movement turns out to lead to a great endpoint (which we said is a generic method); but diffusion provides a great convenience here. We know how $x_t$ is corrupted from $x_0$, in other words, $x_0$ exerts an analytic "pull" for the given noised state $x_t$. 'What makes a **good $x_t$, a noised middle state that is more compatible to those high-reward clean images?**'.

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

Here $\mathbb{E}_Q[v \mid x_t]$ refers to the average velocity of the proposal policy (that collected those training trajectories) given noise image $x_t$, which 0814 abbreviates to $v_\text{base}$.

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

**To sum up, 0814 identified two different value gradient estimators.** The stochastic estimator identifies which Brownian kick in the sampling process created the good image, while the deterministic estimator pulls the model velocity directly to align with the those that generated great endpoints.

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

Now compare this with our deterministic value-gradient estimator (Eq. 31):

$$
\widehat{\nabla V}^{\mathrm{det}}
=
-\frac{1-t}{t}
A(x_0)(v-v_{\mathrm{old}}).
$$

**And the same reward-times-velocity residual appears**: $A(x_0)(v-v_{\mathrm{old}}).$

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
> If the ELBO weight satisfies $w(t)=(1-t)/t$, this is exactly the local variation in Eq. 42. In spirit, we can say AWM is an instance of the deterministic value gradient estimator. (Note a small discrepancy here with the empirical claim of $w(t)=1$ works well: we will explain it later in Section 2.)

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

### 2.1 First order isn't enough: examples

Section 1 has mainly discussed about the question of 'where to move' *around the proposal*, but most RL algorithms are famously 'not purely on-policy'. That is, real algorithms do not take infinitesimal steps; they minimize a loss over many updates. Once finite steps matter, the quadratic part of the loss matters.

We have already seen one example without paying much attention to it. In Eqs. 39-40, we differentiated AWM's squared matching objective at $v_\theta=v_{\mathrm{old}}$. What disappeared when we took that derivative?

Let $\Delta v_\theta=v_\theta-v_{\mathrm{old}}$, $d=v-v_{\mathrm{old}}$, and $s(t)=(1-t)/t$. The signed matching surrogate discussed in Section 1 expands as

<a id="eq-awm-expansion"></a>

$$
\begin{aligned}
\mathcal S_{\mathrm{AWM}}
&=\mathbb E_{x_0,c,t,\epsilon}\left[A w(t)\|v_\theta-v\|^2\right]\\
&=\mathbb E\left[
-2A w(t)d^\top\Delta v_\theta
+A w(t)\|\Delta v_\theta\|^2
\right]+C.
\end{aligned}
\tag{43}
$$

Here and below, $C$ denotes a term independent of $\theta$ during the update. The linear term is the reward-correlated residual we identified in Section 1. The quadratic term has zero first derivative at $\Delta v_\theta=0$, which is why it did not appear in that analysis. Its coefficient is signed: for a negative advantage, this individual matching loss rewards moving farther from the target.

On the other hand [DiffusionNFT](https://arxiv.org/abs/2509.16117)'s objective also looks like regression, but it makes a different choice away from the proposal. Use $\rho\in[0,1]$ for NFT's normalized reward ($r$ in the original paper to avoid confusion), and define $A_{\mathrm{NFT}}=2\rho-1$. Let $\beta>0$ be NFT's mixing parameterand its two implicit policies are

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

For one training example, this gives

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

The two branches contribute the same quadratic coefficient and opposite cross terms. Completing the square makes the effect clear:

<a id="eq-nft-target"></a>

$$
l_{\mathrm{NFT}}
=\beta^2\left\|
\Delta v_\theta-\frac{A_{\mathrm{NFT}}}{\beta}d
\right\|^2
+(1-A_{\mathrm{NFT}}^2)\|d\|^2.
\tag{46}
$$

From this equation, the preferred displacement is $A_{\mathrm{NFT}}d/\beta$: toward its conditional velocity when the normalized advantage is positive, and away when it is negative. The quadratic term ensures that the curvature with respect to the predicted velocity is always $2\beta^2 I$.

| Matching surrogate | Linear term | Quadratic coefficient |
| --- | --- | --- |
| AWM | $-2A w(t)d^\top\Delta v_\theta$ | $A w(t)$, signed |
| NFT | $-2\beta A_{\mathrm{NFT}}d^\top\Delta v_\theta$ | $\beta^2>0$ |

**The first-order analysis identifies the reward-correlated residual, but does not determine the quadratic coefficient.** This is the question Section 2 will explore.

### 2.2 The observed linear-quadratic design space

And here is, in my point of view, the main contribution of 0814: an abstraction that organizes these choices into a common training form. Written as a loss to minimize, the template is

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

> The sampler dependence of $\hat w_2$ is suppressed in the notation.

For the deterministic estimator in Eq. 31, $\widehat{\nabla V}^{\mathrm{det}}=-s(t)Ad$. The AWM expansion in [Eq. 43](#eq-awm-expansion) therefore has coefficients $\hat w_1=A w(t)$ and $\hat w_2=2w(t)/s(t)$ under a uniform-time integral convention. For NFT, dividing by the positive constant $\beta^2$ gives

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

So AWM and NFT fit this template by direct algebra, up to constants. The whole design space can be described as $(\eta_t\text{ (the samplr)},v_{\mathrm{base}},\widehat{\nabla V}_t,\hat w_1,\hat w_2)$. The table in 0814 collects several methods in these coordinates:

![table of methods]({{ '/assets/diffusionrl/methodtable.png' | relative_url }})

The columns are respectively:

- $\eta_t$, or more generally the sampler: which distribution supplies the states?
- $v_{\mathrm{base}}$: what defines zero displacement? For a path-ratio comparison, it is the velocity of the proposal process. For the exact forward-estimator identity, it is the conditional mean $\mathbb E[v\mid x_t,c]$ under the endpoint-and-corruption joint distribution. Using $v_{\mathrm{old}}$ for that mean requires the approximation discussed after Eq. 31; the model merely being an older checkpoint does not establish it.
- $\widehat{\nabla V}_t$: how is the local value gradient estimated?
- $\hat w_1$: what is the sign and strength of the quadratic displacement term?
- $\hat w_2$: how strongly is the improvement direction applied?

> A normalization detail when reading the table: a discrete timestep loss and a continuous-time integrand can differ by a bin width. If the contribution of a bin is written as $[\hat w_2\widehat{\nabla V}^{\top}\Delta v_\theta+\hat w_1\|\Delta v_\theta\|^2]\Delta t$, define $a_t=\hat w_1\Delta t$ and $b_t=\hat w_2\Delta t$. Matching this contribution to the unweighted per-example losses above gives

<a id="eq-bin-coefficients"></a>

$$
\begin{array}{c|cc}
& a_t & b_t\\
\hline
\mathrm{AWM} & A w(t) & 2w(t)/s(t)\\
\mathrm{NFT}/\beta^2 & 1 & 2/[\beta s(t)]
\end{array}
\tag{49}
$$

Thus the table's AWM entries for $w(t)=1$ are $\hat w_1=A/\Delta t$ and $\hat w_2=2/[s(t)\Delta t]$; NFT's are $1/\Delta t$ and $2/[\beta s(t)\Delta t]$.

### 2.3 Weighting is part of the algorithm

The common form does not make all choices of coefficients equivalent. In particular, NFT's positive quadratic is closer to the intuitive idea of a displacement penalty than AWM's signed quadratic, (so that's probably why you do not see an extra KL term in NFT while AWM needs one.)

Additionally what works in expectation are not exempt from noisy Monte Carlo updates for algorithms to work well in practical situations. This sheds light on the AWM observation around Eqs. 37-42: simpler weights such as $w(t)=1$ can work better than the ELBO weight $w(t)=s(t)=(1-t)/t$:

To see what is actually being scaled, let $K_\theta(x_t,t)=\partial v_\theta(x_t,t)/\partial\theta$. With the proposal quantities held fixed, a timestep bin contributes

<a id="eq-parameter-gradient"></a>

$$
\Delta t\,K_\theta^\top
\left[
\hat w_2(t)\widehat{\nabla V}_t
+2\hat w_1(A,t)\Delta v_\theta
\right]
\tag{50}
$$

to the parameter gradient. What matters is the **effective weighted estimator**, together with its integration or sampling weight and the network Jacobian, rather than a coefficient viewed in isolation.

For AWM, the ELBO choice gives the sample integrand

<a id="eq-elbo-weighting"></a>

$$
s(t)A\left[
\|\Delta v_\theta\|^2-2d^\top\Delta v_\theta
\right].
\tag{51}
$$

Both terms inherit the explicit $1/t$ near the clean endpoint. This can amplify sampling noise and concentrate updates at small $t$. Setting $w(t)=1$ removes that factor from both terms. In the framework coordinates, $\hat w_2=2/s(t)$ may itself look singular near the opposite endpoint, but its product with $\widehat{\nabla V}^{\mathrm{det}}=-s(t)Ad$ is simply $-2Ad$. NFT similarly gives $-2A_{\mathrm{NFT}}d/\beta$ after normalization, with a constant positive quadratic coefficient.

This is how I read 0814's scale-bounded weighting principle: inspect the combined update coefficients, including the factors already inside the estimator.

So the weights are part of the algorithm. Why, then, does the particular factor $s(t)$ arise in the theoretical derivation at all? This is a good place to return to the quadratic term in Girsanov. It supplies a probabilistic interpretation of one instance of the template. What that interpretation does and does not pin down is the subject of 2.5.

### 2.4 Where the derived quadratic comes from

Section 1 did not need the particular flow-matching SDE: the path-space argument works for any pair of SDEs with the same diffusion coefficient. To read the quadratic in velocity coordinates, we now need the SDE's drift and diffusion. With sampling time $r=1-t$,

$$
\begin{aligned}
\mu_\theta(y_r,r)&=c_r(y_r)-(1+\eta_t)v_\theta(x_t,t),\\
g_r^2&=\frac{2t\eta_t}{1-t},\qquad t=1-r,\\
\mathrm d y_r&=\mu_\theta(y_r,r)\,\mathrm d r+g_r\,\mathrm d W_r.
\end{aligned}
\tag{52}
$$

The only model-dependent thing we will use is the factor $1+\eta_t$. The model velocity enters the drift linearly, and everything in $c_r$ is shared by $P_\theta$ and $Q$. Comparing two processes with the same initial distribution and diffusion schedule, with $Q$ generated by $v_{\mathrm{base}}$,

$$
\Delta\mu_r
=\mu_\theta-\mu_{\mathrm{base}}
=-(1+\eta_t)(v_\theta-v_{\mathrm{base}})
=-(1+\eta_t)\Delta v_\theta.
\tag{53}
$$

For $v_{\mathrm{base}}=v_{\mathrm{old}}$ and $\eta_t=1$, this is $\Delta\mu=-2\Delta v$, the conversion used in Eq. 41.

We now want the log-ratio in velocity coordinates. Girsanov's formula needs the ratio $\Delta\mu_r/g_r$, so let us compute it rather than define anything:

$$
\frac{\Delta\mu_r}{g_r}
=-\frac{1+\eta_t}{\sqrt{2t\eta_t/(1-t)}}\Delta v_\theta
=-\sqrt{\frac{(1+\eta_t)^2(1-t)}{4t\eta_t}}\,\Delta v_\theta.
$$

The scalar in front is the only combination of the SDE parameters that survives. Call it $\sqrt{2\widetilde w_t}$:

$$
\widetilde w_t:=\frac{(1+\eta_t)^2(1-t)}{4t\eta_t},
\qquad
\frac{\Delta\mu_r}{g_r}=-\sqrt{2\widetilde w_t}\,\Delta v_\theta,
\qquad
\frac12\left\|\frac{\Delta\mu_r}{g_r}\right\|^2
=\widetilde w_t\|\Delta v_\theta\|^2.
\tag{54}
$$

Substituting into Eq. 11 gives the log-ratio in velocity coordinates:

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

Read the two terms. The first is the noise-exploration term: it asks whether the drift change aligns with the Brownian increments. The second is the quadratic term. It is **not** a Taylor expansion of the reward. It is the energy of the proposal gap—the price of $Q$ differing from $P_\theta$. That is why its coefficient is $\widetilde w_t$, and why it vanishes when $\Delta v_\theta=0$.

For $\eta_t=1$, $\widetilde w_t=s(t)$. The large coefficient near $t=0$ reflects the small sampling noise there: a drift change becomes more distinguishable relative to that noise. (The derivation assumes $\eta_t>0$; an ODE sampler with $\eta_t=0$ has to be converted into an SDE first.)

Now the part I got wrong on my first read of 0814. I will show it because the mistake is the point. I read Prop. 1 as a statement about the objective, replaced $e^{M_1}$ by $1+M_1$, and got the surrogate

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

**This is not 0814's derivation. It gives the right first-order term for the wrong reason, and it is not valid as an objective-level statement.** It is, however, the most natural wrong reading: the one the notation $=_\theta$ in Prop. 1 invites. I keep it because the place where it breaks is exactly where the quadratic's status becomes ambiguous. Section 2.5 will show where.

0814's actual route is different, but—this is the point I want to make—fundamentally similar. Prop. 1 is stated as an equality of losses up to a $\theta$-independent constant, but its proof establishes the equality of gradients in the limit $\epsilon\to0$. The statement is stronger than the proof. Under the trust-region condition $|\exp(M_1(\theta^-))-1|<\epsilon$, we have $e^{M_1}\approx1$, so

$$
-\mathcal L_{\mathrm{policy}}(\theta)
=-\mathbb E_Q[A]-\mathbb E_Q[A M_1(\theta)]+O(M_1^2),
$$

and dropping the constant and the $O(M_1^2)$ term gives the same $\mathcal S_{\log}$ as Eq. 56. The difference is the justification: the expansion is local, not a Taylor expansion of the objective. The same martingale identity used in Eq. 18 rewrites the stochastic term as $-\mathbb E_Q\int\nabla V_r^\top\Delta\mu_r\,\mathrm d r$, and changing variables gives

$$
\mathcal S_{\log}(\theta)
=\mathbb E_Q\int_0^1
\left[
(1+\eta_t)\nabla V_t^\top\Delta v_\theta
+A\widetilde w_t\|\Delta v_\theta\|^2
\right]\,\mathrm d t.
\tag{57}
$$

This rewriting is exact for $\mathcal S_{\log}$; $\mathcal S_{\log}$ itself is only locally faithful to the objective. It gives one instance of [Eq. 47](#eq-unified-surrogate), with $\hat w_2=1+\eta_t$ and $\hat w_1=A\widetilde w_t$. Substituting the deterministic estimator at $\eta_t=1$ recovers the coefficients in [Eq. 51](#eq-elbo-weighting).

0814 calls the linear and quadratic components the "on-policy" and "off-policy" terms. The naming is about the proposal, not about the objective: the quadratic vanishes when $v_{\mathrm{old}}=v_\theta$, and appears when the update leaves the proposal. The first term supplies the local improvement signal. The second is the price of the displacement.

> Before reward weighting, the quadratic is a nonnegative path energy. After multiplication by signed $A$, it is not necessarily a positive control cost.

That observation is the crack in the story. The derivation is a **gradient** statement: it tells us how the surrogate's gradient at the proposal relates to the policy gradient. But the paper's framing goes further. It calls the quadratic an "off-policy regulariser" and says it "anchors" $v_\theta$ to $v_{\mathrm{base}}$. That is a claim about the term's **function**, not just its gradient. 0814 never explicitly discusses curvature or the Hessian, but this is a curvature-like claim. The next section asks whether the derivation supports it.

### 2.5 The derivation fixes the gradient, not the curvature

The derivation above is a first-order expansion in $M_1$. It gives the gradient of the surrogate at the proposal. It does not give the objective's second-order term. 0814 never explicitly discusses curvature or the Hessian, so this is not a claim about a missing equation. It is a claim about a word: the paper calls the quadratic an "off-policy regulariser" and says it "anchors" $v_\theta$ to $v_{\mathrm{base}}$. A quadratic regulariser is a penalty on displacement, and the coefficient of that penalty is a curvature-like object. The question is whether the derivation gives the quadratic enough structure to do that job.

To count orders cleanly, forget the velocity parameterization and perturb the drift along a fixed adapted field $h_r$:

$$
\mu_\lambda=\mu_Q+\lambda h,
\qquad
Z:=\int_0^1\frac{h_r^\top}{g_r}\,\mathrm dW_r,
\qquad
C_h:=\int_0^1\frac{\|h_r\|^2}{g_r^2}\,\mathrm d r,
\qquad
M_\lambda=\lambda Z-\frac{\lambda^2}{2}C_h.
\tag{58}
$$

Everything on the right is evaluated along the fixed proposal $Q$; $\lambda$ is the perturbation size. For the flow-matching case, $h_r=-(1+\eta_t)\Delta v_\theta$, so $Z$ is exactly the stochastic integral in Eq. 55 and $C_h$ is twice the quadratic term. Nothing is lost in the abstraction.

Assume the likelihood ratios are valid and the moments permit expansion under the expectation. The true likelihood ratio and the surrogate expand as

$$
e^{M_\lambda}
=1+\lambda Z+\frac{\lambda^2}{2}(Z^2-C_h)+o(\lambda^2),
\qquad
1+M_\lambda
=1+\lambda Z-\frac{\lambda^2}{2}C_h.
\tag{59}
$$

The first line is the exponential; the second is the object we actually kept. For a fixed advantage $A$,

$$
\begin{aligned}
J(\lambda)
&=J(0)+\lambda\mathbb E_Q[AZ]
+\frac{\lambda^2}{2}\mathbb E_Q[A(Z^2-C_h)]
+o(\lambda^2),\\
\widetilde J(\lambda)
:=\mathbb E_Q[A(1+M_\lambda)]
&=J(0)+\lambda\mathbb E_Q[AZ]
-\frac{\lambda^2}{2}\mathbb E_Q[AC_h].
\end{aligned}
\tag{60}
$$

The first-order terms agree by construction. The second-order terms do not. Subtracting,

$$
\widetilde J(\lambda)-J(\lambda)
=-\frac{\lambda^2}{2}\mathbb E_Q[AZ^2]+o(\lambda^2).
$$

The surrogate is missing $\lambda^2\mathbb E_Q[AZ^2]/2$, and that term is of the **same order** as the quadratic energy it does keep. This is the "sneaking in a second-order term" move in its simplest form: we keep a term of order $\lambda^2$ while only justifying the expansion to order $\lambda$. A first-order story with a second-order object. It is a Taylor expansion truncated at first order, with the $x^2$ term left in.

Ito isometry is what makes the mistake easy to miss. Without the reward weight,

$$
\mathbb E_Q[Z^2-C_h]=0,
\qquad\text{but generally}\qquad
\mathbb E_Q[A(Z^2-C_h)]\ne0.
\tag{61}
$$

The unweighted quadratic has zero mean, which is why the energy term looks canonical. The reward weight breaks the cancellation: $A$ is a function of the terminal $x_0$, which depends on the whole path, including the Brownian increments that make up $Z$. So $A$ and $Z^2$ are correlated, and the term that would have cancelled the energy no longer does.

The gap is not an artifact of a particular reward. A centered quadratic reward already shows it. Let $\mathrm dY_r=\lambda\,\mathrm d r+\mathrm dW_r$, $Y_0=0$, and $A(y)=y^2-1$. Under the zero-drift proposal, $Z=W_1$, $C_h=1$, and $\mathbb E_Q[A]=0$. Yet

$$
J(\lambda)=\mathbb E[(W_1+\lambda)^2-1]=\lambda^2,
\qquad
\mathbb E_Q[A(Z^2-C_h)]
=\mathbb E[(W_1^2-1)^2]=2.
\tag{62}
$$

The true objective has a nonzero second-order term; the log-ratio surrogate is flat: both $\mathbb E_Q[AZ]$ and $\mathbb E_Q[AC_h]$ vanish. This is the cleanest demonstration that the surrogate's quadratic is not the objective's second-order term—there is no reward baseline to blame.

A constant reward shows the opposite failure. If the fixed function multiplying the likelihood ratio is $A\equiv a\ne0$, then

$$
J(P_\theta)=a,
\qquad
\mathcal S_{\log}(\theta)
=-a\mathbb E_Q[M_1(\theta)]
=aD_{\mathrm{KL}}(Q\|P_\theta).
\tag{63}
$$

The expected reward is flat, while the surrogate retains KL-like curvature. Here the omitted $Z^2$ term cancels the energy term exactly in the true objective. If advantages are centered, a constant reward gives $A=0$ and both objectives are flat; the point of this diagnostic is that the log surrogate's curvature depends on an additive reward baseline, whereas the true reward does not. The centered example in Eq. 62 shows that the gap also exists without a nonzero baseline.

So what survives? The gradient does. Prop. 1 is a correct statement about the gradient at the proposal, under a trust-region condition. The theorem is not wrong. What is wrong is the step from that theorem to the prose: from "the quadratic appears in a gradient representation" to "the quadratic is an off-policy regulariser that anchors the model." The first is a statement about a derivative. The second is a statement about the term's function—a curvature-like claim. The derivation does not establish the second.

The slide is easy to make because the template is written as a loss. Once the quadratic sits inside a loss, it looks like part of the objective. But the derivation only matched gradients; an object-level equality would need the $Z^2$ term. That is the step from "gradient agreement" to "exact form" that the prose makes and the proof does not. The route is different from my wrong derivation in 2.4; the structure of the slide is the same.

This is a defensible error rather than a fatal one, because 0814 relaxes the coefficients into a design space anyway. Eq. 18 treats $\hat w_1,\hat w_2$ as arbitrary weight functions; the derived values are one instantiation, not a necessity. Once the coefficients are relaxed, the quadratic's job is whatever the designer wants it to be. The practical stakes are low. But the story still matters, because it shapes how people read the quadratic: as a penalty with a principled curvature, rather than as the price of leaving the proposal. It also shapes how people design weights: if the coefficient is a derived curvature, deviating from it looks like a hack; if it is a proposal-gap penalty, deviating from it is a design choice.

**The honest reading is the second one.** The quadratic is the energy of the proposal gap. Its coefficient is a design choice; the derived value is a default. This is why the scale-bounded principle is a design constraint rather than a consequence of the derivation, and why $w(t)=1$ is not a discrepancy. It is a different, bounded trust region.

> This does not invalidate 0814's framework; it explains why the framework has a free $\hat w_1$. Once the quadratic is read as the price of leaving the proposal rather than the curvature of the reward, bounding it is a stability requirement, not a derived identity.


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

### 3.4 Empirical analysis

I am actually thinking of a comparison between DPO KTO (CaPO/IPO) NFT and possibly more, an analogy to the huggingface blog in early 2024 that compared DPO KTO and IPO.




> Ends here.

<!-- {% endcapture %}{% include math-markdown.html content=math_content %} -->
