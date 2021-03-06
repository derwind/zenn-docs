---
title: "Trotter ã®ç©å¬å¼"
emoji: "ð"
type: "tech" # tech: æè¡è¨äº / idea: ã¢ã¤ãã¢
topics: ["math", "Qiskit", "ãã¨ã "]
published: true
---

# ç®ç

ããéå­ã³ã³ãã¥ã¼ã¿ã®æç®ã§ [ãªã¼ã»ãã­ãã¿ã¼ç©å¬å¼](https://ja.wikipedia.org/wiki/%E3%83%AA%E3%83%BC%E3%83%BB%E3%83%88%E3%83%AD%E3%83%83%E3%82%BF%E3%83%BC%E7%A9%8D%E5%85%AC%E5%BC%8F) ãåºã¦ããã®ã§ãç´ã¨éç­ã§æ®´ãæ¸ãã§æ¤è¨¼ãã¦éãã§æ¾ç½®ãã¦ãããQiskit textbook [Proving Universality](https://qiskit.org/textbook/ch-gates/proving-universality.html) ãèª­ãã§ããã¨ãããã£ã¨å¬å¼ãä½¿ããã¦ããã®ã§ãè¡ãå ´ãç¡ããã¦ããæ¾ç½®ã¡ã¢ãè¨äºã«ãããã¨ã«ããã

# æ¦è¦

ãã«ãã«ãç©ºéä¸ã®ä¸è¬ã«ã¯éæçãªèªå·±å±å½¹ä½ç¨ç´  $A$, $B$ ã«ã¤ãã¦ã$A+B$ ã å®ç¾©å $\mathcal{D}(A+B) := \mathcal{D}(A) \cap \mathcal{D}(B)$ ä¸ã§æ¬è³ªçã«èªå·±å±å½¹ã§ããå ´åã«ã**$A$ ã¨ $B$ ãéå¯æã§ãã£ã¦ã**

$$
\begin{align*}
\mathrm{s}\!-\!\lim_{n\to\infty} (e^{it \frac{A}{n}} e^{it \frac{B}{n}})^n = e^{i t (A+B)}
\end{align*}
$$

ãæç«ããã»ã»ã»ã¨ããã®ããTrotter ã®ç©å¬å¼ã§ãããã»ã»ã»ããã¯æµç³ã«å¤§è¢è£ãããã®ã§ãæç® [RS1] ã«å¾ãå½¢ã§ **Lie ã®ç©å¬å¼**ãè¦ããéå­ã³ã³ãã¥ã¼ã¿ã®ã³ã³ãã­ã¹ãã§ã¯ããã§ååã«æããããè¦ãã¨ãã£ã¦ãè¨¼æãé·ãã¯ãªãã®ã§ãçºãã¤ã¤æç®ã§ã¯ç«¯æããã¦ããè¨ç®ã®ç´°é¨ãè£ãã¾ãããã¨ããç¨åº¦ã®ãã¨ã§ããã»ã»ã»ã

# Lie ã®ç©å¬å¼

$A$ ã¨ $B$ ã $d$ æ¬¡æ­£æ¹è¡åã¨ããããã®æãä»¥ä¸ãæç«ããã

$$
\begin{align*}
\exp(A + B) = \lim_{n\to\infty}\left[\exp\left(\frac{A}{n}\right) \exp\left(\frac{B}{n}\right)\right]^n
\end{align*}
$$

éå­ã³ã³ãã¥ã¼ã¿ã®æ¬ã¨ãã®å ´åã ã¨ãä¾ãã° $d=2$ ã¨ãã¦ Pauli è¡å $X$, $Z$ ãã¨ã£ã¦ãã¦ã$A=iaX$, $B=ibZ$ ã¨ç½®ãã¦å©ç¨ããã°ãããååã«å¤§ããª $n$ ã®æã«ä¸¡è¾ºã® $n$ ä¹æ ¹ãã¨ã£ã¦ãè¿ä¼¼çã«

$$
\begin{align*}
e^{iaX/n} e^{ibZ/n} \simeq e^{i(aX + bZ)/n}
\end{align*}
$$

ã¨ã§ãã¾ããã¨ããè©±ã«ãªãã

# è¨¼æ

$S_n = \exp\left(\frac{A+B}{n}\right)$, $T_n = \exp\left(\frac{A}{n}\right) \exp\left(\frac{B}{n}\right)$ ã¨ç½®ãã
ç´æ¥è¨ç®ã§ä»¥ä¸ãç¤ºãããã

$$
\begin{align*}
\sum_{m=0}^0 S_1^m (S_1 - T_1) T_1^{-m} &= S_1 - T_1 = S_1^1 - T_1^1, \\
\sum_{m=0}^1 S_2^m (S_2 - T_2) T_2^{1-m} &= (S_2 - T_2)T_2 + S_2(S_2 - T_2) = S_2^2 - T_2^2, \\
\sum_{m=0}^2 S_3^m (S_3 - T_3) T_3^{2-m} &= (S_3 - T_3)T_3^2 + S_3(S_3 - T_3)T_3 + S_3^2(S_3 - T_3) \\
&= S_3^3 - T_3^3, \\
\end{align*}
$$

ããä¸è¬ã«ã

$$
\begin{align*}
\sum_{m=0}^{n-1} S_n^m (S_n - T_n) T_n^{n-1-m} = S_n^n - T_n^n
\tag{1}
\end{align*}
$$

ãæå¾ããããå®éãå·¦è¾ºãç´æ¥å±éããã¨ã

$$
\begin{align*}
&\ \sum_{m=0}^{n-1} S_n^m (S_n - T_n) T_n^{n-1-m} \\
=&\ (S_n \!-\! T_n)T_n^{n-1} \!+\! S_n(S_n \!-\! T_n)T_n^{n-2} \!+\! S_n^2(S_n \!-\! T_n)T_n^{n-3} \!+\! \cdots \!+\! S_n^{n-1}(S_n \!-\! T_n) \\
=&\ (S_nT_n^{n-1} \!-\! T_n^{n}) \!+\! (S_n^2T_n^{n-2} \!-\! S_n T_n^{n-1}) \!+\! (S_n^3T_n^{n-3} \!-\! S_n^2T_n^{n-2}) \!+\! \cdots \!+\! (S_n^{n} \!-\! S_n^{n-1}T_n)
\end{align*}
$$

ã¨ãªãã$i$ çªç®ã®æ¬å¼§ã®åã®é ã¨ $i+1$ çªç®ã®æ¬å¼§ã®å¾ãã®é  ãã­ã£ã³ã»ã«ãåããæ®ãã®ã¯ $- T_n^{n} + S_n^{n}$ ã§ããããã£ã¦ (1) ãä¸è¬ã® $n$ ã®å¯¾ãã¦æç«ãããã¨ãåãã£ãã

$S_n^n - T_n^n$ ãè©ä¾¡ãããããã®ããã«ãã¾ãã¯ $X = \frac{A}{n}$, $Y = \frac{B}{n}$ ã¨ããã¦ $S_n - T_n$ ã®è©ä¾¡ãè¡ãã

$$
\begin{align*}
&\ \sum_{m=0}^\infty \frac{1}{m!} (X+Y)^m - \left(\sum_{m=0}^\infty \frac{1}{m!} X^m\right) \left(\sum_{m=0}^\infty \frac{1}{m!} Y^m\right) \\
\leq&\ \left\{\! I \!+\! (X\!+\!Y) \!+\! \frac{1}{2} (X\!+\!Y)^2 \!+\! \cdots \!\right\} - \left(\! I \!+\! X \!+\! \frac{1}{2} X^2 \!+\! \cdots \!\right) \left(\! I \!+\! Y \!+\! \frac{1}{2} Y^2 \!+\! \cdots \!\right) \\
\leq&\ \frac{1}{2}(YX - XY) + (\text{3 æ¬¡ä»¥ä¸ã®é })
\tag{2}
\end{align*}
$$

ããã§ãâ3 æ¬¡ä»¥ä¸ã®é â ãè©ä¾¡ããããã®é¨åã¯ä¸è¨ã® Taylor å±éã® 3 æ¬¡ä»¥ä¸ã®é ãªã®ã§ã

$$
\begin{align*}
&\ \|(\text{3 æ¬¡ä»¥ä¸ã®é })\| \\
\leq&\ \left\{\! \frac{1}{3!} \|\!X\!\!+\!\!Y\!\|^3 \!\!+\!\! \frac{1}{4!} \|\!X\!\!+\!\!Y\!\|^4 \!\!+\! \cdots \!\right\} \!+\! \left(\! \|\!X\!\| \!\!+\!\! \frac{1}{2} \|\!X\!\|^2 \!\!+\! \cdots \!\right) \!\! \left(\! \|\!Y\!\| \!\!+\!\! \frac{1}{2} \|\!Y\!\|^2 \!\!+\! \cdots \!\right) \!-\! \| X \| \| Y \| \\
\leq&\ \left\{\! \frac{1}{3!} \frac{\|\!A\!\!+\!\!B\!\|^3}{n^3} \!\!+\!\! \frac{1}{4!} \frac{\|\!A\!\!+\!\!B\!\|^4}{n^4} \!\!+\! \cdots \!\right\} \!+\! \left(\! \frac{\|\!A\!\|}{n} \!\!+\!\! \frac{1}{2} \frac{\|\!A\!\|^2}{n^2} \!\!+\! \cdots \!\right) \!\! \left(\! \frac{\|\!B\!\|}{n} \!\!+\!\! \frac{1}{2} \frac{\|\!B\!\|^2}{n^2} \!\!+\! \cdots \!\right) \!-\! \frac{\| A \| \| B \|}{n^2} \\
\leq&\ \frac{1}{n^3} \!\! \left[ \! \left\{\! \frac{1}{3!} \|\!A\!\!+\!\!B\!\|^3 \!\!+\!\! \frac{1}{4!} \|\!A\!\!+\!\!B\!\|^4 \!\!+\! \cdots \!\right\} \!+\! \left(\! \|\!A\!\| \!\!+\!\! \frac{1}{2} \|\!A\!\|^2 \!\!+\! \cdots \!\right) \!\! \left(\! \|\!B\!\| \!\!+\!\! \frac{1}{2} \|\!B\!\|^2 \!\!+\! \cdots \!\right) \!-\! \| A \| \| B \| \! \right] \\
\leq&\ \frac{1}{n^3} \left\{\exp( \|A+B\|) + \exp \|A\| \exp \|B\| \right\}
\end{align*}
$$

ã¨è©ä¾¡ã§ãããããã (2) ã¨ä½µããã¨ã

$$
\begin{align*}
&\ \left\|\sum_{m=0}^\infty \frac{1}{m!} (X+Y)^m - \left(\sum_{m=0}^\infty \frac{1}{m!} X^m\right) \left(\sum_{m=0}^\infty \frac{1}{m!} Y^m\right) \right\| \\
\leq&\ \frac{1}{n^2} \cdot \frac{1}{2} \|BA-AB\| + \frac{1}{n^3} \left\{\exp( \|X+Y\|) + \exp \|X\| \exp \|Y\| \right\} = \mathcal{O}(1/n^2)
\end{align*}
$$

ã¤ã¾ãã

$$
\begin{align*}
\| S_n - T_n \| = \mathcal{O}(1/n^2)
\tag{3}
\end{align*}
$$

ã¨ããè©ä¾¡å¼ãå¾ãã

ã¾ã

$$
\begin{align*}
\|S_n^m\| \leq \|S_n\|^m \leq \|S_n\|^n \leq \exp(\|A+B\|) \leq C
\tag{4}
\end{align*}
$$

$$
\begin{align*}
\|T_n^{n-1-m}\| \leq \|T_n\|^{n-1-m} \leq \|T_n\|^n \leq \exp(\|A\|) \exp(\|B\|) \leq C
\tag{5}
\end{align*}
$$

ã¨ããè©ä¾¡ãå¾ãããã®ã§ã(1) ã¨ (3) ã¨ä½µãã¦

$$
\begin{align*}
\| S_n^n - T_n^n \| \leq \sum_{m=0}^{n-1} C^2 \cdot \mathcal{O}(1/n^2) = \mathcal{O}(1/n) \to 0 \quad\text{as}\quad n \to \infty
\end{align*}
$$

$S_n^n = \exp(A+B)$ ã§æéç¢ºå®ã§ããã®ã§ä¸»å¼µãå¾ãã

# ã¾ã¨ã

ãã¾ãåºããªãæ¨ªå¹ã®ä¸­ã«ç¡çç¢çæ°å¼ãè©°ãè¾¼ãã ã®ã§æ±ããªã£ãããæ±ãããå¼ãé©å½ã«åè§£ãã¦ãåå¥ã« $n$ ã«ã¤ãã¦ã©ããããã®ãªã¼ãã¼ã§å¤§ãããªãã®ãå°ãããªãã®ããè©ä¾¡ããã¨ããæ¯è¼çåç´ãªè¨ç®ã§è¨¼æãããã¨ãã§ããã

$A$ ã¨ $B$ ãå¯æã§ããå ´åã$\exp(A+B) = \exp(A) \exp(B)$ ãæç«ããã®ã§ç¹ã«ä½ãé©ããã¨ã¯ãªãã®ã ããéå¯æã§ãã£ã¦ãåæ§ã®çµæãæç«ããã®ã§èå³æ·±ããæ¯è¼çå¤å¸çãªå®çã ã¨æããããããããã£ããã®ãéå­ã³ã³ãã¥ã¼ã¿ã®ã²ã¼ãã®è¿ä¼¼ã«ä½¿ãããã®ã¯é¢ç½ãã[^1]

[^1]: ã²ã¼ãã®è¿ä¼¼ã¨ããç¹ã§ã¯ Solovay-Kitaev ã®å®çãæåã¨æãããããã¡ãã¯ãã¼ã«ã»ã³&ãã£ã³ã«ããã¨ 1995 å¹´ã¨æ¯è¼çè¿å¹´ç¤ºããããã®ã§ããããããLie ã®ç©å¬å¼ãç¤ºãããã®ã¯ãã¤ã®ãã¨ãããããªãããTrotter ã®ç©å¬å¼ã¯ 1959 å¹´ã¨ãªããªãå¤ãã

# æç®

[RS1] M. Reed and B. Simon. Methods of Modern Mathematical Physics, I. Functional Analysis, pp.295, Academic Press, New York, 1981
