
# Laminify - Instantly classify data with [Lamini](lamini.ai) & Llama 2

Train a new classifier with just a prompt.

`./train.sh --class "cat: CAT_DESCRIPTION" --class "dog: DOG_DESCRIPTION"`

`./classify.sh 'woof'`

```
{'data': 'woof',
 'prediction': 'dog',
 'probabilities': array([0.37996491, 0.62003509])}
```

For example, here is a cat/dog classifier trained using prompts.

Cat prompt:

```
Cats are generally more independent and aloof. Cats are also more territorial and may be more aggressive when defending their territory.
Cats are self-grooming animals, using their tongues to keep their coats clean and healthy. Cats use body language and vocalizations,
such as meowing and purring, to communicate.  An example cat is whiskers, who is a cat who lives in a house with a human.
Another example cat is furball, who likes to eat food and sleep.  A famous cat is garfield, who is a cat who likes to eat lasagna.
```

Dog prompt:
```
Dogs are social animals that live in groups, called packs, in the wild. They are also highly intelligent and trainable.
Dogs are also known for their loyalty and affection towards their owners. Dogs are also known for their ability to learn and
perform a variety of tasks, such as herding, hunting, and guarding. Dogs are also known for their ability to learn and
perform a variety of tasks, such as herding, hunting, and guarding.  An example dog is snoopy, who is the best friend of
charlie brown.  Another example dog is clifford, who is a big red dog.
```

# Installation

Clone this repo, and run the `train.sh` or `classify.sh` command line tools.  

Requires docker: https://docs.docker.com/get-docker 

Setup your lamini keys (free): https://lamini-ai.github.io/auth/

`git clone git@github.com:lamini-ai/laminify.git`

`cd laminify`

Train a new classifier.

```
./train.sh --help

usage: train.py [-h] [--class CLASS [CLASS ...]] [--train TRAIN [TRAIN ...]] [--save SAVE] [-v]

options:
  -h, --help            show this help message and exit
  --class CLASS [CLASS ...]
                        The classes to use for classification, in the format 'class_name:prompt'.
  --train TRAIN [TRAIN ...]
                        The training data to use for classification, in the format 'class_name:data'.
  --save SAVE           The path to save the model to.
  -v, --verbose         Whether to print verbose output.

```

Classify your data.

```
./classify.sh --help

usage: classify.py [-h] [--data DATA [DATA ...]] [--load LOAD] [-v] [classify ...]

positional arguments:
  classify              The data to classify.

options:
  -h, --help            show this help message and exit
  --data DATA [DATA ...]
                        The training data to use for classification, any string.
  --load LOAD           The path to load the model from.
  -v, --verbose         Whether to print verbose output.

```

These command line scripts just call python inside of docker so you don't have to care about an environment.  

If you hate docker, you can also run this from python easily...


# Python Library

Install it
`pip install lamini`

Instantiate a classifier

```
from lamini import LaminiClassifier

# Create a new classifier
classifier = LaminiClassifier()
```

Define classes using prompts

```
classes = { "SOME_CLASS" : "SOME_PROMPT" }

classifier.prompt_train(classes)
```

Add some training examples (optional)

```
data = ["example 1", "example 2"]
classifier.add_data_to_class("SOME_CLASS", data)

# Don't forget to train after adding data
classifier.train()
```

Classify your data

```
# Classify the data
prediction = classifier.predict(data)

# Get the probabilities for each class
probabilities = classifier.predict_proba(data)
```

Save your model

```
classifier.save("SOME_PATH")
```

Load your model
```
classifier = LaminiClassifier.load(args["load"])
```

# FAQ

## How does it work?

Laminify converts your prompts into a pile of data, using the Llama 2 LLM. It then finetunes another LLM to distinguish between each pile of data.  

We use several specialized LLMs derived from Llama 2 to convert prompts into piles of training examples for each class.  The code for this is available
in the lamini python package if you want to look at it.  Working on open sourcing it when I'm not too distracted...

## Why wouldn't I just use a normal classifier like BART, XGBoost, BERT, etc?

You don't need to label any data using Laminify.  Labeling data sucks.

No fiddling with hyperparameters. Fiddle with prompts instead.  Hopefully english is easier than attention_dropout_pcts.

## Why wouldn't I just use a LLM directly?

A classifier can only output a class.  An LLM might answer the question "Is this talking about a cat" with "Well... that depends on ....".  Writing a parser sucks.

Added benefit: classifiers give you probabilities and can be calibrated: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

## Why does this FAQ sound so sarcastic?

Because it is 5am

