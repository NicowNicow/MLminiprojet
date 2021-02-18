# Prise en main d'OpenNN

## _Apprentissage de la base de données MNIST_
<br>


Nous allons voir ici un example d'utilisation de la librairie OpenNN, appliquée à la base de donnée [MNIST](http://yann.lecun.com/exdb/mnist/)

Cette base de données est un classique du machine learning, elle contient 60000 images en niveaux de gris, de tailles 28x28 pixels, et contenant des chiffres manuscrits allant de 0 à 9. Le but est de faire apprendre ces échantillons au système, puis de vérifier l'efficacité du modèle sur 10000 autres échantillons de validation.

Commençons par charger les dépendances nécessaires (noter que le chemin vers *opennn.h* est à modifier au besoin):

```Cpp
#include <iostream>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;
```

Chargons à présent la database, qui a été ici compilée en fichier *mnist_train.csv*:

```cpp
cout << "OpenNN. MNIST Example." << endl;

srand(static_cast<unsigned>(time(nullptr)));

// Data set

DataSet data_set("data/mnist_train.csv", ',', false);

data_set.set_input();
data_set.set_column_use(0, OpenNN::DataSet::VariableUse::Target);
data_set.numeric_to_categorical(0);
data_set.set_batch_instances_number(5);

const Vector<size_t> inputs_dimensions({1, 28, 28});
const Vector<size_t> targets_dimensions({10});
data_set.set_input_variables_dimensions(inputs_dimensions);
data_set.set_target_variables_dimensions(targets_dimensions);

const size_t total_instances = 100;
data_set.set_instances_uses((Vector<string>(total_instances, "Training").assemble(Vector<string>(60000 - total_instances, "Unused"))));
data_set.split_instances_random(0.75, 0, 0.25);
```

On retrouve bien les dimensions 28x28 des images. On remarquera que la première dimension définie est 1, ceci est dû au fait que les images sont uniquement en niveaux de gris - il y a donc un seul channel de couleur.
