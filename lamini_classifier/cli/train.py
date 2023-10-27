from lamini import LaminiClassifier

import argparse

from pprint import pprint


def main():
    """This is a program that trains a classifier using the LaminiClassifier class.

    LaminiClassifier is a powerful classifier that uses a large language model to classify data.

    It has the ability to define classes, each using a prompt, and then classify data based on that prompt.

    It can also be trained on examples of data for each class, and then classify data based on that training.

    """

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Parse the class names and prompts using the format "class_name:prompt"
    parser.add_argument(
        "--class",
        type=str,
        nargs="+",
        action="extend",
        help="The classes to use for classification, in the format 'class_name:prompt'.",
        default=[],
    )

    # Parse the training data using the format "class_name:data"
    parser.add_argument(
        "--train",
        type=str,
        nargs="+",
        action="extend",
        help="The training data to use for classification, in the format 'class_name:data'.",
        default=[],
    )

    # Parse the path to save the model to
    parser.add_argument(
        "--save",
        type=str,
        help="The path to save the model to.",
        default="models/model.lamini",
    )

    # Parse the path to save the model to
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The path to save data checkpoints.",
        default="models/checkpoint.jsonl",
    )

    # Parse verbose mode
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print verbose output.",
        default=False,
    )

    # conert the arguments to a dictionary
    args = vars(parser.parse_args())

    # Create a new classifier
    classifier = LaminiClassifier(saved_examples_path=args["checkpoint"])

    # Train the classifier on the training data
    for class_data in args["train"]:
        class_name, data = class_data.split(":")
        classifier.add_data_to_class(class_name, [data])

    # Train the classifier on the classes
    classes = {}

    for class_prompt in args["class"]:
        class_name, prompt = class_prompt.split(":")
        assert class_name not in classes, f"Class name '{class_name}' already exists."
        classes[class_name] = prompt

    if len(classes) == 0 and len(args["train"]) == 0:
        classes = get_default_classes()

    if args["verbose"]:
        pprint("Training on classes:")
        pprint(classes)

    classifier.prompt_train(classes)

    if args["verbose"]:
        pprint(classifier.get_data())

    # Save the classifier to the path
    classifier.save(args["save"])


def get_default_classes():
    """Returns a dictionary of default classes to use for training.

    The default classes are:
        - "cat"
        - "dog"
    """

    print(
        "WARNING ------ No classes or data were specified, using default cat vs dog classes."
    )

    return {
        "cat": "Cats are generally more independent and aloof. Cats are also more territorial and may be more aggressive when defending their territory.  Cats are self-grooming animals, using their tongues to keep their coats clean and healthy. Cats use body language and vocalizations, such as meowing and purring, to communicate.  An example cat is whiskers, who is a cat who lives in a house with a human.  Another example cat is furball, who likes to eat food and sleep.  A famous cat is garfield, who is a cat who likes to eat lasagna",
        "dog": "Dogs are social animals that live in groups, called packs, in the wild. They are also highly intelligent and trainable. Dogs are also known for their loyalty and affection towards their owners. Dogs are also known for their ability to learn and perform a variety of tasks, such as herding, hunting, and guarding. Dogs are also known for their ability to learn and perform a variety of tasks, such as herding, hunting, and guarding.  An example dog is snoopy, who is the best friend of charlie brown.  Another example dog is clifford, who is a big red dog.",
    }


main()
