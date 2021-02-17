# Machine Learning Mini Projet

Mini-projet de tests de librairies de Machine Learning sur Python & Unix.  Le groupe est composé de Florian SANANES (OpenNN) et Nicolas MENGOZZI (Theano).

## Videos de présentation

[Vidéo de présentation de OpenNN](https://github.com/NicowNicow/MLminiprojet/blob/main/vids/openNN_pitch.mp4)

Vidéo de présentation de Theano:

<a href="https://www.youtube.com/watch?v=AYyUuKGogbw">
   <img src="https://img.youtube.com/vi/AYyUuKGogbw/0.jpg">
</a>

<br>

## Installation

Afin d'installer les deux librairies utilisées dans ce projet, [OpenNN](https://www.opennn.net/) et [Theano](https://www.tutorialspoint.com/theano/index.htm), il suffit d'executer le [script d'installation](https://github.com/NicowNicow/MLminiprojet/blob/14a55f329b63c61fb16873534a3756683c6b9133/install_script.sh), situé à la racine du repository, dans le répertoire de travail choisi: `install_script.sh`


---

## Historique des librairies

<h3><u>OpenNN</h3></u>
<br>

> Le développement de OpenNN s’initia au Centre Internacional de Métodos Numéricos en Ingeniería (CIMNE), en 2003, en faisant partie d’un projet de recherche de l’Union européenne appelé ‘RAMFLOOD’. Ensuite, il continua en faisant partie d’autres projets similaires. Actuellement, OpenNN est développé par l’entreprise startup Artelnics. En 2014, la page «Big Data Analytics Today» qualifia OpenNN comme le numéro 1 dans la liste de projets d’intelligence artificielle inspirés du fonctionnement du cerveau. Dans la même année, ce logiciel fut sélectionné parmi les 5 meilleures applications d’exploration de données, par «ToppersWorld».  
Source: [Wikipédia](https://fr.wikipedia.org/wiki/OpenNN)

<br>

<h3><u>Theano</h3></u>
<br>

> Theano is an open source project primarily developed by the Montreal Institute for Learning Algorithms (MILA) at the Université de Montréal.
The name of the software references the ancient philosopher Theano, long associated with the development of the golden mean.
On 28 September 2017, Pascal Lamblin posted a message from Yoshua Bengio, Head of MILA: major development would cease after the 1.0 release due to competing offerings by strong industrial players. Theano 1.0.0 was then released on 15 November 2017.
On 17 May 2018, Chris Fonnesbeck wrote on behalf of the PyMC development team that the PyMC developers will officially assume control of Theano maintenance once they step down.  
Source: [Wikipédia](https://en.wikipedia.org/wiki/Theano_(software))

---

## Caractéristiques et points forts/faibles des librairies

<h3><u>Caractéristiques d'OpenNN</h3></u>
<br>

Avantages:

- La librairie est aisée d'utilisation si l'on est familier avec le C++.
- La documention sur l'utilisation est complète et claire.

Inconvénients:

- La procédure d'installation est défectueuse.
- OpenNN nécessite d'être *linkée* à d'autres librairies afin d'accéder à de bonnes performances. Cet aspect ne semble pas documenté.
  
<br>

<h3><u>Caractéristiques de Theano</h3></u>
<br>

Avantages:

- La librairie est compatible avec d'autre librairies, telles que numpy.
- La librairie permet de générer simplement des [Graphs de calculs](https://www.tutorialspoint.com/theano/theano_computational_graph.htm) représentant l'intégralité du programme.
- Ces graphs sont très pertinents pour le développement de réseaux de neurones récurrents.
- Des wrappers tels que Keras et Lasagne sont disponibles pour faciliter le développement.

Inconvénients:

- Les modèles les plus larges sont parfois long à compiler.
- La librairie est plus lourde que ses concurrents (pyTorch, ...).
- Les messages d'erreurs sont parfois confus.
- Le support sur les modèles pré-entrainé laisse à désirer.
- La librairie ne supporte pas l'usage de GPU en SLI ou de multi-GPU.
- Comme le montre l'existence de wrappers, la librairie de base peut s'avérer difficile à utiliser.
- La librairie serait problématique à utiliser sur un déploiement AWS (A vérifier).
  
---
  
## Prise en main des librairies

<u><h3> Prise en main d'OpenNN</h3></u>
<br>

ToDo Florian

<br>

<u><h3> Prise en main de Theano: Exemple de création d'un réseau de neurones double couche</h3></u>
<br>

Afin d'appréhender la prise en main de la librairie avec un exemple concret, nous allons ici créer un réseau de neurones à deux couches, dont le but est l'apprentissage de la fonction logique XNOR.
On rappelle tout d'abord la table de vérité de la fonction XNOR:  

| Entrée 1 | Entrée 2 | Sortie |
|:-:|:-:|:-:|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |
<br>

Nous allons ici réaliser un apprentissage supervisé d'un réseau à plusieurs couches. Ce réseau est défini de la façon suivante:

<br>

<img src="./img/theano-diagram-nn.png">

<br>
<br>

De ce diagramme, on peut définir la table de vérité en fonction des neurones:

| Entrée 1 | Entrée 2 | Neurone 1 | Neurone 2 | Neurone 3 | Sortie |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 1 | 1 |
<br>

L'étude préliminaire étant terminée, nous allons désormais pouvoir passer à l'implémentation. Le code source au complet est disponible dans le fichier: `theano_test/xnor_logical_function_neural.py`  

Commençons par importer les différentes librairies nécessaires à la création d'un réseau de neurones.

```python
# Imports
import theano
import theano.tensor
from theano import function

from random import random
import numpy
```

Afin d'utiliser Théano, il est nécessaire d'importer la librairie. On importe également theano.tensor, qui permet l'accès à des types (matrix, scalar, ...) et des méthodes de calculs (exp, dot, log, ...) compatibles avec theano. Les functions theano sont également nécessaire au bon fonctionnement du projet. Enfin, on importe numpy et random, afin de faciliter la manipulation de données.  

```python
# Variables Definition
inputMatrix = theano.tensor.matrix('inputMatrix')

weight1 = theano.shared(numpy.array([random(), random()]))
weight2 = theano.shared(numpy.array([random(), random()]))
weight3 = theano.shared(numpy.array([random(), random()]))

bias1 = theano.shared(1.)
bias2 = theano.shared(1.)

learningRate = 0.01
iterationNumber = 100000
```

Définissons les variables qui vont nous être nécessaires. Tout d'abord, l'entrée de notre réseau de neurones est une matrice de dimension deux. Elle contient les différents couples de valeurs possibles.  
Ensuite, on définit plusieurs variables de type shared. Ces variables sont partagées par les différents appels des fonctions Théano, et permettent de convertir directement depuis des types Python. On définit ici les poids correpondants aux trois neurones, ainsi que les deux biais des deux couches.
(Plus d'infos sur les types ajoutés par Théano [ici](https://theano.readthedocs.io/en/rel-0.6rc3/library/tensor/basic.html))  
Nous définissons également le taux d'apprentissage en float, ainsi qu'un entier représentant le nombre d'itérations pour l'apprentissage.

```python
# Variables Definition
neuron1 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(inputMatrix, weight1) - bias1))
neuron2 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(inputMatrix, weight2) - bias1))
firstLayerResulMatrix = theano.tensor.stack([neuron1, neuron2], axis=1)

neuron3 = 1/(1 + theano.tensor.exp( -theano.tensor.dot(firstLayerResulMatrix, weight3) - bias2))
```

Il est maintenant temps de définir nos neurones. pour chacune des neurones, on utilise la formule suivante:  

<br>
<img src="img/expression.png"/>

<br>
Notons qu'il nous est alors nécessaire de définir une matric temporaire entre les deux couches du réseau de neurone, afin de pouvoir appliquer la même formule à la troisième neurone.

```python
# Gradient Definition
realOutput = theano.tensor.vector('realOutput')
cost = -(realOutput*theano.tensor.log(neuron3) + (1 - realOutput)*theano.tensor.log(1-neuron3)).sum()
gradWeight1, gradWeight2, gradWeight3, gradBias1, gradBias2 = theano.tensor.grad(cost, [weight1, weight2, weight3, bias1, bias2])
```

Nous définissons ici un vector, nommé realOutput, qui correspond à l'output attendu à la sortie du réseau. Nous créons également la formule de calcul du coût pour ce réseau de neurones. Enfrin, grâce à cette formule, nous calculons les gradients des différents poids et des deux biais.

```python
# Weight and Bias update
TrainingFunction = function(
    inputs = [inputMatrix, realOutput],
    outputs = [neuron3, cost],
    updates = [
        [weight1, weight1-learningRate*gradWeight1],
        [weight2, weight2-learningRate*gradWeight2],
        [weight3, weight3-learningRate*gradWeight3],
        [bias1, bias1-learningRate*gradBias1],
        [bias2, bias2-learningRate*gradBias2]
    ]
)
```

Ici, nous définissons la fonction principale du projet. Il s'agit d'une Théano function. Nous lui donnons deux entrées, ici la matrice d'entrée ainsi que la sortie attendue. Nous lui donnos également les sorties attendues: le résultat du calcul de la troisième neurone, ainsi que le coût de l'itération.
Enfin, on définit les updates des différents poids et biais.

```python
# Inputs and Outputs Definition
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [1, 0, 0, 1]
```

Nous définissons ici la matrice d'input initiale, qui est associée à l'objet inputMatrix déclaré précédemment. Comme il s'agit d'un entrainement supervisé, nous déclarons également les résultats attendus pour chaque couple de valeurs.

```python
# Model Training
costArray = []
for index in range(iterationNumber):
    predictionArray, iterationCost = TrainingFunction(inputs, outputs)
    costArray.append(iterationCost)
```

C'est ici que débute l'entrainement de notre réseau. On définit une costArray, qui nous permettra de conserver le coût de cha que opération. Ce n'est pas une étape obligatoire, mais il peut être utile de conserver le coût afin d'étudier l'entrainement de notre réseau.
Ensuite, on applique la Theano function, que nous avons défini précedemment, à notre matrice d'entrée, autant de fois qu'indiqué par notre variable iterationNumber.

```python
# Output Printing
print('Here are the outputs of the Neural Network:')
for index in range (len(inputs)):
    print('The output for the input [%d, %d] is %.2f' % (inputs[index][0], inputs[index][1], predictionArray[index])) 
```

Après écriture du script d'affichage et exécution du fichier python, on obtient le résultat suivant dans la console:

<img src="img/resultat_entrainement.jpg">

Nous avons ainsi appris les bases de l'utilisation de Théano, et savons désormais entrainer un réseau de neurone à plusieurs couches.

---

## Sources

<h3><u>Sources pour la documentation d'OpenNN</h3></u>
<br>

ToDo Florian

<h3><u>Sources pour la documentation Theano</h3></u>
<br>
  
- [JournalDev - Tutorial de développement Theano](https://www.journaldev.com/17840/theano-python-tutorial)
- [Theano - API Documentation](https://theano-pymc.readthedocs.io/en/latest/library/index.html)
- [DVLUP - Pro and Cons of AI Frameworks](https://dvlup.tech/2018/12/18/ai-frameworks-pros-cons/)
- [INRIA - Deep Learning Framework](https://project.inria.fr/deeplearning/files/2016/05/DLFrameworks.pdf)
- [Quora - Strengths of Theano vs Torch](https://www.quora.com/Deep-Learning-What-are-the-strengths-of-Theano-vs-Torch)
- [RecodeMinds - The Ultimate FaceOff between different Deep Learning Algorithm](https://recodeminds.com/blog/the-ultimate-face-off-between-different-deep-learning-frameworks/)
- [Edureka - Theano vs TensorFlow](https://www.edureka.co/blog/theano-vs-tensorflow/)
