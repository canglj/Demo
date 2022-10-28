# EM算法

## 原理分析

### 一个简单的例子

【例1】 $X$ 服从四点分布：$\left(\begin{array}{ll} y_1 & y_2 & y_3 & y_4 \\ (\frac{1}{2}-\frac{\theta}{4}) & (\frac{1-\theta}{4}) & (\frac{1+\theta}{4}) & (\frac{\theta}{4})\end{array}\right)$

写出似然函数之后可以没有办法求出参数的解析解。比如说$L(\theta)=(\frac{1}{2}-\frac{\theta}{4})^{y_1}(\frac{1-\theta}{4})^{y_2}(\frac{1+\theta}{4})^{y_3}(\frac{\theta}{4})^{y_4}$

可以引入一个隐变量。把$(\frac{1}{2}-\frac{\theta}{4})$拆成两部分，比如说拆成$\frac{1-\theta}{4}$和$\frac{1}{4}$,然后取值为第一部分的记为$z_1$,剩下的就是$y_1-z_1$，其他几个也同理，这样的话最后可以合并一些东西。

【例2】 比如抛硬币问题，有两枚硬币$A, B$，质地可能不均匀，不知道每次抛的是什么硬币，只知道正反面结果，然后估计每个硬币正面朝上的概率。

【例3】 假如我们需要调查学校的男生和女生的身高分布，我们抽取随机抽取200个人，统计他们的身高，但是不说明是男生还是女生，但是我们知道他们各自服从正态分布，如何估计他们的参数$\mu,\sigma$,以及怎么判断是男生的身高还是女生的。

### 理论推导

==EM: Expectation Maximization==：

解决存在**隐变量**（也就是不可观测的变量，比如说上面那个例2，选取哪个硬币就是不可观测的）时的优化问题。每次迭代都包括两步，一个是期望步（$E$步），二个是极大步（$M$步）

==Jensen 不等式==：

假设 $f(x)$ 为下凸函数，则有 $E(f(x))\geq f(E(X))$

观测到的随机变量$X$的样本为$X=(x_1,x_2,\cdots,x_n)$,隐含变量 $Z=(z_1,z_2,\cdots,z_n)$

$\hat{\theta}=\operatorname{argmax} \sum_{i=1}^n \log p\left(x_i ; \theta\right)=\operatorname{argmax} \sum_{i=1}^n \log \sum_{z_i} p\left(x_i, z_i ; \theta\right)$ (全概率公式，这个要对$\theta$求导非常困难，但是log函数是concave的，我们可以利用Jensen不等式)

令$Q_i$表示隐变量$Z$的分布，于是可以得到：

$\begin{aligned} \sum_{i=1}^n \log \sum_{z_i} p\left(x_i, z_i ; \theta\right) &=\sum_{i=1}^n \log \sum_{z_i} Q_i\left(z_i\right) \frac{p\left(x_i, z_i ; \theta\right)}{Q_i\left(z_i\right)} \\ & \geq \sum_{i=1}^n \sum_{z_i} Q_i\left(z_i\right) \log \frac{p\left(x_i, z_i ; \theta\right)}{Q_i\left(z_i\right)} \end{aligned}$

注：（由于 $\sum_{z_i} Q_i\left(z_i\right) \log \left[\frac{p\left(x_i, z_i \theta\right)}{Q_i\left(z_i\right)}\right]$ 为 $\frac{p\left(x_i, z_i ; \theta\right)}{Q_i\left(z_i\right)}$的期望，并且$\log(x)$是上凸函数，这样利用Jensen不等式，就会出来一个期望，右边那个求和又是关于$z_i$的期望。）

然后等号成立的条件，即$\frac{p\left(x_i, z_i ; \theta\right)}{Q_i\left(z_i\right)}={C}$, ${C}$ 为常数

由于 $Q_i\left(z_i\right)$ 是一个分布，所以满足: $\sum_{z_i} Q_i\left(z_i\right)=1$ ，则 $\sum_{z_i} p\left(x_i, z_i ; \theta\right)=C$ 由上面两个式子，我们可以得到：
$$
Q_i\left(z_i\right)=\frac{p\left(x_i, z_i ; \theta\right)}{\sum_z p\left(x_i, z_i ; \theta\right)}=\frac{p\left(x_i, z_i ; \theta\right)}{p\left(x_i ; \theta\right)}=p\left(z_i \mid x_i ; \theta\right)
$$
这就是一个后验概率。相当于我们要求得$\operatorname{argmax} \sum_{i=1}^n \sum_{z_i} Q_i\left(z_i\right) \log \frac{p\left(x_i, z_i ; \theta\right)}{Q_i\left(z_i\right)}$，然后不断地迭代。去掉常数部分，实际上就是要极大化对数似然下界：$\arg \max _\theta \sum_{i=1}^n \sum_{z_{i}} Q_i\left(z_{i}\right) \log p\left(x_{i}, z_{i} ; \theta\right)$

右边那个求和可以理解为$\log p\left(x_{i}, z_{i} ; \theta\right)$关于$Z$的期望

接下来要证明==EM算法的收敛性==，也就是$\theta^{(t)}$对应的$L(\theta)$会不断增加。

令 $L\left(\theta, \theta_j\right)=\sum_{i=1}^n \sum_{z_{i}} p\left(z_{i} \mid x_{i} ; \theta_j\right)\log p\left(x_{i}, z_{i} ; \theta\right)$

$H\left(\theta, \theta_j\right)=\sum_{i=1}^n \sum_{z_{i}} p\left(z_{i} \mid x_{i} ; \theta_j\right) \log p\left(z_{i} \mid x_{i} ; \theta\right)$

相减得：

$\sum_{i=1}^n \log P\left(x_{i} ; \theta\right)=L\left(\theta, \theta_j\right)-H\left(\theta, \theta_j\right)$

于是

$\begin{aligned} \sum_{i=1}^n \log p\left(x_{i} ; \theta_{j+1}\right) &-\sum_{i=1}^n \log p\left(x_{i} ; \theta_j\right)=\left[L\left(\theta_{j+1}, \theta_j\right)-L\left(\theta_j, \theta_j\right)\right]  -\left[H\left(\theta_{j+1}, \theta_j\right)-H\left(\theta_j, \theta_j\right)\right] \end{aligned}\geq 0$

## 算法流程

输入: 观察到的数据 $x=\left(x_1, x_2, \ldots x_n\right)$ ，联合分布 $p(x, z ; \theta)$ ，条件分布 $p(z\mid x, \theta)$ ，最大迭代次数$J$。
算法步骤:
(1) 随机初始化模型参数 $\theta$ 的初值 $\theta_0$ 。
(2) $j=0,2, \ldots, J$ 开始EM算法迭代:

- $\mathrm{E}$ 步：计算联合分布的条件概率期望:
$$
\begin{aligned}
&Q_i\left(z_i\right)=p\left(z_i \mid x_i, \theta_j\right) \\
&L\left(\theta, \theta_j\right)=\sum_{i=1}^n \sum_{z_i} Q_i\left(z_i\right) \log {p\left(x_i, z_i ; \theta\right)}
\end{aligned}
$$
- $\mathrm{M}$ 步：极大化 $L\left(\theta, \theta_j\right)$, 得到 $\theta_{j+1}$ :
  $\theta_{j+1}=\operatorname{argmax} L\left(\theta, \theta_j\right)$
- 如果 $\theta_{j+1}$ 已经收敛，则算法结束。否则继续进行$E$步和$M$步进行迭代。

输出: 模型参数 $\theta$ 。

问题：初始值如何选取？





## 实例1（混合正态分布）

假设统计男生女生的平均身高，但是不知道统计的是男还是女，男生和女生的身高分别服从正态分布。

$X_i | Z_i=1 \sim N(\mu_1,\Sigma_1),X_i | Z_i=2 \sim N(\mu_2,\Sigma_2)$

如果数据图是多峰的，基本上可以用EM算法，尤其是数据没有标记的时候。机器学习 top 10 算法之一。还有 MCMC 算法。

EM算法怎么用到图像处理，还有 HMM 算法。

## 实例2（抛硬币）

