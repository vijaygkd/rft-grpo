# rft-grpo
Reinforcement Learning Fine-Tuning with GPRO

## RFT - Reinforcement Fine-Tuning

Ref course - https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/


Goals:
- Implement GRPO Reinforcement FT on LLM for reasoning fine-tuning.




## Mathematical Formulation

The GRPO (Generalized Relative Policy Optimization) loss function is given by:

$$
\boxed{\;
\mathcal L(\theta)=
-\underbrace{\mathbb{E}_{(x,a)\sim\pi_{\text{old}}}\!\big[\,r_\theta(x,a)\,A(x,a)\,\big]}_{\displaystyle\text{policy term}}
+\beta\;\underbrace{\mathrm{KL}\!\bigl(\pi_\theta\,\|\,\pi_{\text{ref}}\bigr)}_{\displaystyle\text{regulariser}}
\;}
$$

Where:
- $\mathcal L(\theta)$ is the total loss function
- $r_\theta(x,a)$ is the reward function parameterized by $\theta$
- $A(x,a)$ is the advantage function
- $\pi_{\text{old}}$ is the old policy distribution
- $\pi_\theta$ is the current policy parameterized by $\theta$
- $\pi_{\text{ref}}$ is the reference policy
- $\beta$ is the regularization coefficient
- $\mathrm{KL}$ denotes the Kullback-Leibler divergence

The first term represents the policy optimization objective, while the second term acts as a regularizer to prevent the policy from deviating too far from the reference policy.

