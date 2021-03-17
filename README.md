Deep sur graph
==============

Liste des articles et résumés
-----------------------------

### Graph Machines and Their Applications to Computer-Aided Drug Design: a New Approach to Learning from Structured Data : Goulon et al (2006)
Un des premiers modèles de réseaux de neurones pour traiter des graphes (des molécules ici).

On détermine pour un graphe donné, un arbre couvrant ayant comme racine un nœud central du graphe. Les cycles sont donc supprimés mais l'information (de leur existance) est conservée dans les features attachés aux nœuds.
On fait ensuite remonter l'information dans l'arbre jusqu'au nœud racine central avec un réseau (fully connected). Le partage des paramètre réside dans le fait que c'est le même réseau qui est utilisé
à chaque fois.

Article dans le cadre d'une thèse : Une nouvelle méthode d’apprentissage de données structurées : applications à l’aide à la découverte de médicaments, Goulon 2008.
Premier article : From Hopfield nets to recursive networks to graph machines: Numerical machine learning for structured data (Goulon et al, 2005)

**Output :** Valeur pour chaque graphe (nombre de cycles, diamètre, Wiener index)
**Pourquoi cet article ?** Un des premiers articles sur les réseaux de neurones pour les graphes, qui utilise directement la structure du graphe (à travers l'utilisation d'un arbre couvrant)


### The Graph Neural Network Model : Scarselli et al. 2009 
Un des premiers modèle de Graph Neural Network (GNN). Des états cachés sont associés à chaque noeud. Ces noeuds sont mis à jours en fonction de leurs états caché, les états cachés de leurs vosins, les attributs de leurs voisins et les attributs de leur arcs adjacents. Pour cela, un message est d’abord calculé :

$$x_n = f_w(l_n, l_{co[n]},x_{ne[n]}, l_{ne[n]})$$


La fonction $f_w​$ est utilisée de manière itérative jusqu’à convergence du message. Le message permet ensuite de prendre une décision sur le noeud actuel.

$$o_n = g_w(x_n, l_n)$$

Ici, $g_w​$ correspond à un RNN et $f_w$ à un MLP.

Ce premier modèle présente plusieurs limitations, dont la plus importante est le fait que $f_w$​ doit être une carte de contraction afin d’assurer la convergence.

**Output :** Prédiction sur les noeuds, ou sur les graphes en ajoutant un noeud spécial.\
 **Pourquoi cet article ?** Un des premier article à parler de  GNN et à les théoriser, avec Gori et al (2005)

### Gated Graph Sequence Neural Network : Li et al. 2015 
Le modèle de Scarselli est contraignant sur de nombreux points :

1.  Hypothèses fortes (carte de contractions).
2.  RNN c’est bien, GRU c’est mieux.
3.  Pas de décision séquentielle.

Utilisation de la BPTT et donc d’un nombre d’itérations fixe plutôt que les cartes de contraction.
 Plus de séparation entre état caché et attribut. Les états caché sont initialisés via les attributs.
 L’état caché des noeuds est du coup mis à jour à chaque itération, via une GRU.  Le calcul du message se fait de la manière suivante :

$$a_v^{(t)} = A_{v:}^T[h_1^{(t-1)T}\dots h_{|V|}^{(t-1)T}]^T + b$$

avec $A \in \mathbb{R}^{D|V| \times 2D|V|}$ et $b$ les paramètres du réseau de neurone dépendamment des noeuds entrants et sortants. Bien que non expliqué dans l’article, le labelling des arc est codé en réalisant un réseau différent selon l’arc (attributs discrets), et en augmentant ainsi dimension de $A \in \mathbb{R}^{E \times D|V| \times 2D|V|}$ 

Utilisation de deux GG-NN en cascade afin de prendre une décision séquentielle : un pour prédire la sortie et un pour mettre à jour les états cachés.

**Output :** Sur les noeuds, via une softmax, sur les graphes via un embedding $h_G$​, ou séquentiel.
$$h_G = tanh(\sum\limits_{v \in V} \sigma(i(h_v^(T), x_v) \odot tanh(j(h_v^{(T)}, x_v)))$$

**Pourquoi cet article ?** Fait la transition entre les GNN de Scarselli et le deep learning moderne. Propose un embedding de graphe. Les GG-NN sont encore utilisé.

### Learning Graphical State Transition : Johnson, 2017 
Propose une évolution des GG-NN et des GGS-NN, les Gated Graph Transformer Neural Network (GGT-NN). Souhaite utiliser les graphes comme représentation intermédiaire dans des problèmes de questions et réponses, cette représentation intermédiaire servant de mémoire, avec un fonctionnement un peu similaire aux Neural Turing Machine. Il s’agit donc d’un problème de génération de graphe.

Modifie légèrement la définition d’un graphe afin de la rendre différentiable. Le graphe est ainsi définit comme suit : $G=(V,C)$ avec $V$ l’ensemble des noeuds $v$ et
$C \in \mathbb{R}^{|V| \times |V| \times Y}$ une matrice de connectivité avec $Y$ l’ensemble des types d’arcs possibles.

Les attributs des noeuds sont représenté sous forme vectorielle, avec pour contrainte $\sum\limits_{j=1}^N x_{v,j} = 1$, avec $N$ le nombre d’attribut possible.

Ajout également d’une *strength*, représentant la croyance que le noeud
devrait exister $s_v = [0,1]$

Finalement, les valeurs de C sont également des *strength*. Chaque arc
est ainsi représenté par une valeur entre 0 et 1.

**Output :** Noeud, graphe, séquence
 **Pourquoi cet article ?** La représentation $C$ qui remplace la
matrice d’adjacence est utilisée implicitement dans beaucoup d’articles,
sans que ce soit précisé.

### Neural Message Passing for Quantum Chemistry : Gilmer et al. 2017 
Cet article apporte plusieurs aspects intéressants. Tout d’abord, il
propose une généralisation des GNN (sous le terme de Message Passing
Neural Network (MPNN)). Ils définissent ainsi une fonction message
$$m_v^{t+1} = \sum\limits_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$$ ainsi qu’une fonction de mise à jour $$h^{t+1} = U_t(h_v^t, m_v^{t+1})$$ avec $M_t$​ et $U_t$​ des fonctions quelconques. Ils représentent ainsi 8 modèles différents de la littérature sous cette forme générale.

* **Convolutional Networks for Learning Molecular Fingerprints, Duvenaud et al. (2015).**
$$M_t(h_v^t, h_w^t, e_{vw}) = (h_w, e_{vw}) \text{(concaténation
    simple)}$$    
    $$U_t(h_v^t, m_v^{t+1}) = \sigma(H_t^{deg(v)}, m_v^{t+1})$$ avec     $H_t^{deg(v)}$​ un réseau de neurones fonction du degré de $v$.
    
*   **Gated Graph Neural Networks, Li et al. (2015).**
    $$M_t(h_v^t, h_w^t, e_{vw}) = A_{e_{vw}}(h_w^t)$$ avec $A_{e_{vw}}$​​ le réseau  fonction du type d’arc.
    $$U_t(h_v^t, m_v^{t+1}) = GRU(h_v^t, m_v^{t+1})$$


*  **Interaction Networks, Battaglia et al. (2016)**
   $$M_t(h_v^t, h_w^t, e_{vw}) = NN(h_v^t, h_w^t, e_{vw})$$
   $$U_t(h_v^t, m_v^{t+1}) = NN(h_v^t, x_v^t, m_v^{t+1})$$ avec $x_v^t$​ un vecteur représentant les *influences extérieurs* sur le noeud $v$.
*   **Molecular Graph Convolutions, Kearnes et al. (2016)** 
Utilise des états cachés sur les arcs plutôt que sur les noeuds $$M_t(h_v^t, h_w^t, e_{vw}) = e^t_{vw}$$
    $$U_t(h_v^t, m_v^{t+1}) = \alpha (W_1(\alpha (W_0h_v^t), m_v^{t+1}))$$ avec     $\alpha$ une ReLu et $W_1, W_0$ deux NN.
    $$e_{vw}^{t+1} = \alpha (W_4(\alpha (W_2e_{vw}^t), \alpha(W_3(h_v^t, h_w^t))))$$

* **Deep Tensor Neural Networks, Schutt et al. (2017)**
   $$M_t(h_v^t, h\_w^t, e_{vw}) = \text{tanh}(W^{fc}((W^{cf}h_w^t+b_1)\odot (W^{df}e_{vw} + b_2)))$$
    $$U_t(h_v^t, m_v^{t+1}) = h_v^t + m_v^{t+1}$$

Les auteurs incluent également les méthodes basées sur la théorie
spectrale des graphes dans leur généralisation

*  **Laplacian Based Methods, Bruna et al. (2013); Defferrard et al.  (2016)** 
    $$M_t(h_v^t, h_w^t, e_{vw}) = C^t_{v,w}h_w^t $$avec $C$ la matrice des vecteurs propres du Laplacien du graphe.
    $$U_t(h_v^t, m_v^{t+1}) = \sigma(m_v^{t+1})$$
    
* **Semi-Supervised Classification with Graph Convolutional Networks. Kipf & Welling. (2016)** 
$$M_t(h_v^t, h_w^t, e_{vw}) = c_{v,w}h_w^t $$
    avec $c_{vw} = (\text{deg(v)deg(w)})^{−1/2} A_{vw}$.
    $$U_t(h_v^t, m_v^{t+1}) = ReLU(W^t, m_v^{t+1})$$

En plus de la généralisation de ces 8 modèles, les auteurs présentent une grande base de graphes (100k+) pour la chimie quantique qui peut être très intéressante pour AGAC.

Les auteurs présentent également une alternative à l’embedding de graphe présenté dans Gated Graph Neural Network. basé sur une projection linéaire itérée MMM fois, qui semble plus efficace que celui proposée précédemment.

**Output :** Principalement par graphe, le dataset cherchant à prédire 13 propriétés différentes sur les graphes.

**Pourquoi cet article** L’article suivant présente une généralisation plus importante, mais l’apport du dataset peut être très intéressant. De plus, la présentation des différents modèle sous la forme généralisé eest cool.

### Relational inductive biases, deep learning, and graph network.  DeepMind, Google Brain, MIT, University of Edimburgh (2018)

27 auteurs des 4 affiliations précédentes proposent (entre autre) une généralisation plus importante des Graph Neural Network, notamment en ajoutant des états cachés $e_k$​ aux arcs également.
 On met ainsi à jour les états cachés des arc ($e\_k$​), des noeuds ($v_i$​) et de l’état général du graphe ($u$) de la manière suivante :

-   $e'_k = \phi^e(e_k, v_{r_k}, v_{s_k}, u)$

-   $v'_i = \phi^v(\bar{e}'_i, v_{i}, u)$

-   $u' = \phi^u(\bar{e}',\bar{v}', u)$

	 Avec $\phi$ des réseaux de neurones. Les $\bar{e}'_i$, $\bar{e}'$ et $\bar{v}'$ correspondent à l’agrégation des différents arcs ou noeuds via une fonction invariante aux permutations (max, moyenne, somme…)

-   $\bar{e}'_i = \rho^{e \rightarrow v}(E'_i)$

-   $\bar{e}' = \rho'^{e \rightarrow u}(E')$

-   $\bar{v}' = \rho^{v \rightarrow u}(V')$

Avec $E'_i$​ l’ensemble des arcs adjacents au noeud i, $E'$ l’ensemble des arc et $V′$ l’ensemble des noeuds.

-   $E_i = \{(e_k, r_k, s_k)\}_{r_k=i, k=1:N_e}$​

-   $V' = \{v'_i\}_{i=1:N_v}$

-   $E' = \bigcup_iE'_i$
- Il y a beaucoup d’autres points abordés dans cet article, mais ce simple point rend l’article suffisamment intéressant pour être lu.

**Output :** Noeud ou graphe via $u$
 **Pourquoi cet article ?** Généralisation la plus récente et la plus
complète surement actuellement.




Coarsening / Pooling
--------------------

Méthodes utilisées pour le coarsening de graphe dans les réseaux sur graphes.

### Weighted Graph Cuts without Eigenvectors A Multilevel Approach. Dhillon et al (2007)

Algorithme greedy multi-échelle permettant d'approcher certaines objectives comme le min-cut.

Utilisé dans :

* Defferard et al (2016)
* ...


### Kron Reduction of Graphs With Applications to Electrical Networks. Dorfler and Bullo (2013)

Utilisé dans :

* Simmonovsky et al (2017)
* ...

