# Machine Learning Mini Projet

Mini-projet de tests de librairies de Machine Learning sur Python & Unix.  Le groupe est composé de Florian SANANES (OpenNN) et Nicolas MENGOZZI (Theano).


## Videos de présentation

[Vidéo de présentation de OpenNN](https://github.com/NicowNicow/MLminiprojet/blob/main/vids/openNN_pitch.mp4)


Vidéo de présentation de Theano:

[![Vidéo de présentation de Theano](https://img.youtube.com/vi/AYyUuKGogbw/0.jpg)](https://www.youtube.com/watch?v=AYyUuKGogbw)

## Installation

Afin d'installer les deux librairies utilisées dans ce projet, [OpenNN](https://www.opennn.net/) et [Theano](https://www.tutorialspoint.com/theano/index.htm), il suffit d'executer le [script d'installation](https://github.com/NicowNicow/MLminiprojet/blob/14a55f329b63c61fb16873534a3756683c6b9133/install_script.sh), situé à la racine du repository, dans le répertoire de travail choisi:

``` sh install_script.sh ```

   ---

## Historique des librairies

### OpenNN

> Le développement de OpenNN s’initia au Centre Internacional de Métodos Numéricos en Ingeniería (CIMNE), en 2003, en faisant partie d’un projet de recherche de l’Union européenne appelé ‘RAMFLOOD’. Ensuite, il continua en faisant partie d’autres projets similaires. Actuellement, OpenNN est développé par l’entreprise startup Artelnics. En 2014, la page «Big Data Analytics Today» qualifia OpenNN comme le numéro 1 dans la liste de projets d’intelligence artificielle inspirés du fonctionnement du cerveau. Dans la même année, ce logiciel fut sélectionné parmi les 5 meilleures applications d’exploration de données, par «ToppersWorld».  
Source: [Wikipédia](https://fr.wikipedia.org/wiki/OpenNN)

### Theano

> Theano is an open source project primarily developed by the Montreal Institute for Learning Algorithms (MILA) at the Université de Montréal.
The name of the software references the ancient philosopher Theano, long associated with the development of the golden mean.
On 28 September 2017, Pascal Lamblin posted a message from Yoshua Bengio, Head of MILA: major development would cease after the 1.0 release due to competing offerings by strong industrial players. Theano 1.0.0 was then released on 15 November 2017.
On 17 May 2018, Chris Fonnesbeck wrote on behalf of the PyMC development team that the PyMC developers will officially assume control of Theano maintenance once they step down.  
Source: [Wikipédia](https://en.wikipedia.org/wiki/Theano_(software))

---

## Caractéristiques et points forts/faibles des librairies

### Caractéristiques d'OpenNN

Avantages:
- La librairie est aisée d'utilisation si l'on est familier avec le C++.
- La documention sur l'utilisation est complète et claire.

Inconvénients:
- La procédure d'installation est défectueuse.
- OpenNN nécessite d'être *linkée* à d'autres librairies afin d'accéder à de bonnes performances. Cet aspect ne semble pas documenté.

### Caractéristiques de Theano

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

### Prise en main d'OpenNN

ToDo

### Prise en main de Theano
  
ToDo
