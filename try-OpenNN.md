# Prise en main d'OpenNN

## _Apprentissage de la base de données MNIST_
<br>


Nous allons voir ici un example d'utilisation de la librairie OpenNN, appliquée à la base de donnée [MNIST](http://yann.lecun.com/exdb/mnist/)

Cette base de données est un classique du machine learning, elle contient 60000 images en niveaux de gris, de tailles 28x28 pixels, et contenant des chiffres manuscrits allant de 0 à 9. Le but est de faire apprendre ces échantillons au système, puis de vérifier l'efficacité du modèle sur 10000 autres échantillons de validation. Ici, seuls les échantillons d'apprentissage seront utilisés.

Commençons par charger les dépendances nécessaires (noter que le chemin vers *opennn.h* est à modifier au besoin):

```Cpp
#include <iostream>
#include <time.h>

#include "../../opennn/opennn.h"
using namespace OpenNN;
```

Chargons à présent la database, qui a été ici compilée en fichier *mnist_train.csv*:

```cpp
cout << "OpenNN. MNIST Example." << endl;

srand(static_cast<unsigned>(time(nullptr))); // seeding the RNG.

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

On retrouve bien les dimensions 28x28 des images. On remarquera que la première dimension définie est 1, ceci est dû au fait que les images sont uniquement en niveaux de gris - il y a donc un seul channel de couleur. D'autre part, la dimension de sortie est 10. En effet, il n'y a que 10 classes à discriminer, les chiffres de 0 à 9. D'un autre côté, on remarque que les échantillons vont être mélangés, puis séparés en deux paquets - le premier, contenant 75% des samples, seront utilisé pour l'apprentissage, et l'autre pour la validation. Enfin, la taille des paquets d'échantillons qui vont être utilisés dans la descente de gradient a été fixée à 5.

Considérons à présent la construction du réseau de neurones multicouche utilisé:

```cpp
// Scaling layer

ScalingLayer* scaling_layer = new ScalingLayer(inputs_dimensions);
neural_network.add_layer(scaling_layer);

const Vector<size_t> scaling_layer_outputs_dimensions = scaling_layer->get_outputs_dimensions();

// Convolutional layer 1

ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(scaling_layer_outputs_dimensions, {8, 5, 5});
neural_network.add_layer(convolutional_layer_1);

const Vector<size_t> convolutional_layer_1_outputs_dimensions = convolutional_layer_1->get_outputs_dimensions();

// Pooling layer 1

PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_1_outputs_dimensions);
neural_network.add_layer(pooling_layer_1);

const Vector<size_t> pooling_layer_1_outputs_dimensions = pooling_layer_1->get_outputs_dimensions();
```

Tout d'abord un *ScalingLayer* a été utilisé afin de s'assurer que les données d'entrées soient bien aux bonnes dimensions. Puis une couche de convolution a été définie par *ConvolutionalLayer*, composée de 8 filtres/kernels, chacun de taille 5x5. Enfin, une couche de pooling a été utilisée, afin de réduire la taille des *feature maps* données en entrée de la couche suivante. Nous ne le montrerons pas ici, mais deux autres couches de convolution-pooling ont été utilisées, la syntaxe est rigoureusement identique, seules les dimensions auront changé - elles seront respectivement de 4x3x3 et 2x3x3.

```cpp
// Perceptron layer

PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3_outputs_dimensions.calculate_product(), 18);
neural_network.add_layer(perceptron_layer);

const size_t perceptron_layer_outputs = perceptron_layer->get_neurons_number();

// Probabilistic layer

ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
neural_network.add_layer(probabilistic_layer);

neural_network.print_summary();
```

Ensuite, une couche de neurones totalement connectés *PerceptronLayer* a été définie, contenant 18 neurones. Elle est suivie d'un *ProbabilisticLayer* servant a émettre les probabilités prédites pour chacune des 10 classes.

```cpp
// Training strategy

TrainingStrategy training_strategy(&neural_network, &data_set);
training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

StochasticGradientDescent* sgd_pointer = training_strategy.get_stochastic_gradient_descent_pointer();

sgd_pointer->set_minimum_loss_increase(1.0e-6);
sgd_pointer->set_maximum_epochs_number(12);
sgd_pointer->set_display_period(1);
sgd_pointer->set_maximum_time(1800);

const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();
```

On remarque à ce stade que la stratégie d'apprentissage choisie a été la descente de gradient stochastique (par paquets). De plus, la fonction de coût choisie est l'erreur quadratique moyenne. Enfin, le nombre d'époques durant lesquelles l'apprentissage va se dérouler a été fixé à 12. Augmenter cette quantité peut améliorer le score final, tant que l'on s'assure de ne pas faire de sur-apprentissage.

```cpp
// Testing analysis

TestingAnalysis testing_analysis(&neural_network, &data_set);

Matrix<size_t> confusion = testing_analysis.calculate_confusion();

cout << "\n\nConfusion matrix: \n" << endl << confusion << endl;
cout << "\nAccuracy: " << (confusion.calculate_trace()/confusion.calculate_sum())*100 << " %" << endl << endl;
```

On ici évalué la qualité de la reconnaissance du réseau de neurones, en affichant sa *précision*, ainsi que sa matrice de confusion.

```cpp
// Save results

data_set.save("data/data_set.xml");

neural_network.save("data/neural_network.xml");
neural_network.save_expression("data/expression.txt");

training_strategy.save("data/training_strategy.xml");
training_strategy_results.save("data/training_strategy_results.dat");
```

Finalement, le réseau de neurones post apprentissage peut être sauvegardé, ainsi aussi que les différents paramètres utilisées lors dudit apprentissage.
