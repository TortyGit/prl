PRL Template
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



