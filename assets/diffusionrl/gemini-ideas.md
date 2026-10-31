
### Visuals for Section 1: Path-Space & Value-Gradient Estimators

#### 1. The "Many Paths to $x_0$" Funnel (Section 1.1 & 1.2, near Eq. 3–4)

* **Type:** Custom conceptual sketch (Excalidraw / TikZ).
* **What it solves:** Gives immediate visual intuition for why $\nabla_\theta \log P_\theta(x_0)$ is intractable (Eq. 1) and why lifting to trajectory space $\tau$ (Eq. 3–4) fixes it.


* **What to draw:**
* **Left ($t=1$, prior noise $p_1(x_1)$):** A Gaussian cloud of starting points.


* **Middle ($t \in (0,1)$):** Multiple tangled, branching stochastic trajectories $\tau^{(1)}, \tau^{(2)}, \dots$ evolving under $Q(\tau)$, with a zoomed-in magnifying glass on a single discrete step $y_r \to y_{r+\Delta r}$.


* **Right ($t=0$, clean endpoint $x_0$):** Several different paths converging to the same clean image $x_0$, where the scalar terminal reward $A(x_0)$ is evaluated.




* **Caption idea:** Why endpoint likelihood $\log P_\theta(x_0)$ is intractable: infinitely many trajectories $\tau$ can reach $x_0$, forcing us to do importance sampling over the full path $\frac{P_\theta(\tau)}{Q(\tau)}$ instead.



#### 2. Geometry of a Single Girsanov Step (Section 1.2, right after Eq. 9)

* **Type:** Custom vector diagram.
* **What it solves:** Eq. 6–9 is the mathematical backbone of Section 1.2. A simple vector triangle turns the algebra of $\Delta \mu := \mu_\theta - \mu_q$ and $\epsilon_r$ into obvious geometry.


* **What to draw:**
```text
                     y_{r + \Delta r} (actual sampled next state)
                       *
                      / \
    g_r\sqrt{\Delta r}\epsilon_r  /   \  y_{r+\Delta r} - (y_r + \mu_\theta \Delta r)
                    /     \
                   /       \
     y_r *--------*---------*
          \mu_q\Delta r  \Delta\mu\Delta r  (y_r + \mu_\theta \Delta r)

```


* Show state $y_r$ branching into two deterministic drift predictions: $y_r + \mu_q \Delta r$ (the proposal $Q$) and $y_r + \mu_\theta \Delta r$ (the new policy $P_\theta$), separated by the vector $\Delta \mu \Delta r$.


* Draw the realized Brownian noise step $g_r \sqrt{\Delta r}\epsilon_r$ jumping from $y_r + \mu_q \Delta r$ to the actual next state $y_{r+\Delta r}$.


* Visually highlight the angle between $\Delta \mu$ and $\epsilon_r$ (**the linear drift-noise correlation $\frac{\Delta \mu}{g_r}\sqrt{\Delta r}\epsilon_r$**) vs. the length of the gap $\Delta \mu$ (**the quadratic energy cost $-\frac{1}{2}\Vert{}\frac{\Delta \mu}{g_r}\Vert{}^2 \Delta r$**).





#### 3. The Martingale Value Landscape $V_r(y_r)$ (Section 1.3, near Eq. 16–19)

* **Type:** Custom 2D heatmap / contour sketch.
* **What it solves:** Section 1.3 introduces martingales, filtrations $\mathcal{F}_r$, and Itô's lemma ($\mathrm{d}V_r = g_r \nabla V_r^\top \mathrm{d}W_r$). Readers who aren't comfortable with stochastic calculus often get intimidated here.


* **What to draw:**
* A 2D space with horizontal axis $r \in [0, 1]$ (sampling time) and vertical axis $y_r$ (state space).


* Background shading showing the value function $V_r(y_r) = \mathbb{E}_Q[A(y_1) \mid y_r]$: blurry and flat near $r=0$ ($V_0 = 0$), sharpening into high-reward (green) and low-reward (red) regions as $r \to 1$.


* Overlay a trajectory where the deterministic drift $\mu_Q$ moves *along* the contour lines of $V_r$ (zero expected change in $V_r$, i.e., the martingale property $\mathbb{E}[\mathrm{d}V_r]=0$), while the random Brownian kicks $\mathrm{d}W_r$ push the trajectory *across* the contours in the direction of $\nabla_{y_r} V_r$.





#### 4. $\widehat{\nabla V}^{\text{sto}}$ (Flow-GRPO) vs. $\widehat{\nabla V}^{\text{det}}$ (AWM / NFT) (Section 1.4 & 1.5 summary)

* **Type:** Adapted from **0814 Figure 1** + **DiffusionNFT Figure 3** (or a side-by-side custom diagram).
* **What it solves:** This is the most important conceptual contrast in Section 1. You can either screenshot/adapt **Figure 3 of the DiffusionNFT paper** (which shows how $v_\theta$ is pushed away from $v_{\text{old}}$ along the velocity residual toward high-reward samples) or draw a two-panel side-by-side comparison:


* **Left Panel — Stochastic Estimator ($\widehat{\nabla V}^{\text{sto}}$, Flow-GRPO):** Show a multi-step reverse SDE rollout $x_1 \to \dots \to x_t \to x_{t-\Delta t} \to \dots \to x_0$. At step $t$, highlight the tiny random Brownian kick $\epsilon_r$. Show the terminal reward $A(x_0)$ being broadcast all the way back to ask: *"Did this random wiggle $\epsilon_r$ cause $x_0$ to be good?"*


* **Right Panel — Deterministic Estimator ($\widehat{\nabla V}^{\text{det}}$, AWM / NFT):** Show only the clean endpoint $x_0$ (with reward $A(x_0)$) being analytically re-noised via the straight-line forward path $x_t = (1-t)x_0 + t\epsilon$. At $x_t$, draw two velocity vectors: the model's average velocity $v_{\text{old}}(x_t, t)$ and the endpoint-specific velocity $v = \frac{x_t - x_0}{t}$. Draw the difference vector $d = v - v_{\text{old}}$, showing how $\widehat{\nabla V}^{\text{det}}$ pulls the policy directly along $A(x_0)(v - v_{\text{old}})$ without needing reverse-SDE step noise!





---

### Visuals for Section 2: Finite Updates, Weighting, & The Second-Order Trap

#### 5. The Quadratic Loss Bowls: AWM vs. DiffusionNFT (Section 2.1, right after the comparison table)

* **Type:** 10-line `matplotlib` plot (or borrowed/adapted from **DiffusionNFT Figure 2/3**).
* **What it solves:** Visually proves why AWM is unstable without an external KL penalty when $A < 0$, while NFT is unconditionally stable.


* **What to plot:** A 1D cross-section of the loss $\mathcal{S}(\Delta v_\theta)$ on the y-axis against the scalar displacement $\Delta v_\theta$ on the x-axis (fixing $d = 1, w(t)=1, \beta=1$):


* **Panel A (Positive Advantage, $A > 0$):** Both AWM ($\mathcal{S}_{\text{AWM}} = -2A\Delta v_\theta + A\Delta v_\theta^2$) and NFT ($l_{\text{NFT}} = -2\beta A_{\text{NFT}}\Delta v_\theta + \beta^2 \Delta v_\theta^2$) are upward-facing convex parabolas (U-shapes) with a well-defined minimum at $\Delta v_\theta > 0$.


* **Panel B (Negative Advantage, $A < 0$):**
* **AWM (dashed red curve):** Flips into an **upside-down parabola ($\cap$-shape)**! Even though its slope at $\Delta v_\theta = 0$ points in the right direction, minimizing the loss sends $\Delta v_\theta \to -\infty$.


* **DiffusionNFT (solid blue curve):** Remains an **upward-facing convex parabola ($\cup$-shape)**, simply shifting its finite minimum to the negative side $\Delta v_\theta^* = \frac{A_{\text{NFT}}}{\beta}d < 0$ (Eq. 46).




* (Bonus: Pair this plot right next to **Figure 3 from the DiffusionNFT paper**, which illustrates the implicit positive/negative policies $v_\theta^+$ and $v_\theta^-$ straddling $v_{\text{old}}$.)





#### 6. The $1/t$ Weight Singularity Plot (Section 2.3, near Eq. 51)

* **Type:** Simple 1D `matplotlib` plot (or adapted from **AWM's paper figures on weighting**).
* **What it solves:** Makes the "scale-bounded weighting principle" in Section 2.3 effortless to grasp.


* **What to plot:** Plot the effective gradient multiplier on the y-axis against diffusion time $t \in (0, 1]$ on the x-axis:


* **Curve 1 (Theoretical ELBO / Girsanov weight $s(t) = \frac{1-t}{t}$):** Skyrockets to $+\infty$ as $t \to 0$ (near the clean image endpoint) and drops to $0$ at $t = 1$. Shade the region $t \in (0, 0.1)$ in red and label it *"Monte Carlo variance explosion zone"*.


* **Curve 2 (Scale-bounded choice $w(t) = 1$, where $\hat w_2 \cdot s(t) = 2$):** A completely flat, stable horizontal line across all $t \in [0, 1]$.





#### 7. The "Missing $Z^2$ Term" Taylor Expansion Diagram (Section 2.5, near Eq. 59–60)

* **Type:** A visual equation / box-cancellation diagram.
* **What it solves:** Section 2.5's critique hinges on a subtle point: why keeping the $O(\lambda^2)$ term inside $M_\lambda$ while truncating $e^{M_\lambda} \approx 1 + M_\lambda$ is a broken second-order expansion.


* **What to draw:** Write out the expansion of $\mathbb{E}_Q[A e^{M_\lambda}]$ in three colored boxes organized by powers of $\lambda$:


* **$O(1)$:** $\mathbb{E}_Q[A]$


* **$O(\lambda)$ (1st Order — Kept):** $+\lambda \mathbb{E}_Q[A Z]$ *(Valid policy gradient!)*


* **$O(\lambda^2)$ (2nd Order — Split):**
* Box 1 (Green, Kept in $\mathcal{S}_{\log}$): $-\frac{\lambda^2}{2}\mathbb{E}_Q[A C_h]$ *(Girsanov quadratic energy)*


* Box 2 (Red, Dropped by $1+M_\lambda$): $+\frac{\lambda^2}{2}\mathbb{E}_Q[A Z^2]$ *(Outer product of Brownian noise)*




* Draw a bracket connecting the two $O(\lambda^2)$ boxes with the callout: Without $A$, $\mathbb{E}_Q[Z^2] = \mathbb{E}_Q[C_h]$ cancels out. With $A(y_1)$, they don't cancel—keeping one without the other breaks the curvature!





#### 8. The Two Toy Counterexamples Plot: $J(\lambda)$ vs. $\widetilde{J}(\lambda)$ (Section 2.5, right after Eq. 62–63)

* **Type:** Two-panel 1D `matplotlib` plot.
* **What it solves:** Turns Eq. 62 and Eq. 63 from dense algebra into an unforgettable visual mic-drop at the end of Section 2.5.


* **What to plot:**
* **Left Panel — Centered Quadratic Reward $A(y) = y^2 - 1$ (Eq. 62):** Plot the true objective $J(\lambda) = \lambda^2$ (a clean upward parabola) vs. the Girsanov log-surrogate $-\mathcal{S}_{\log}(\lambda) = 0$ (a dead-flat horizontal line on the x-axis). Label: True objective is curved; surrogate is flat.


* **Right Panel — Constant Baseline Reward $A(y) \equiv a$ (Eq. 63):** Plot the true objective $J(\lambda) = a$ (a dead-flat horizontal line) vs. the Girsanov log-surrogate $-\mathcal{S}_{\log}(\lambda) = a - \frac{a}{2}\lambda^2$ (a curved parabola that bends upward or downward depending on the sign of $a$). Label: True objective is flat; surrogate hallucinates curvature.