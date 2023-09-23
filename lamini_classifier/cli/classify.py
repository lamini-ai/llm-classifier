from lamini import LaminiClassifier

import argparse

from pprint import pprint


def main():
    """This is a program that runs inference using a classifier using the LaminiClassifier class.

    LaminiClassifier is a powerful classifier that uses a large language model to classify data.

    It has the ability to define classes, each using a prompt, and then classify data based on that prompt.

    It can also be trained on examples of data for each class, and then classify data based on that training.

    """

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Parse the data to classify
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        action="extend",
        help="The training data to use for classification, any string.",
        default=[],
    )

    # The user can either specify data with --data or directly on the command line
    parser.add_argument(
        "classify",
        type=str,
        nargs="*",
        action="extend",
        help="The data to classify.",
        default=[],
    )

    # Parse the path to save the model to
    parser.add_argument(
        "--load",
        type=str,
        help="The path to load the model from.",
        default="models/model.lamini",
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

    # Load the model
    classifier = LaminiClassifier.load(args["load"])

    # Get the data to classify
    data = []

    data += args["data"]
    data += args["classify"]

    if len(data) == 0:
        raise Exception("No data to classify.")

    if args["verbose"]:
        print("args data", args["data"])
        print("args classify", args["classify"])
        print("Data to classify:", data)

    # Classify the data
    prediction = classifier.predict(data)

    # Get the probabilities for each class
    probabilities = classifier.predict_proba(data)

    # Print the results
    for i in range(len(data)):
        pprint(
            {
                "data": data[i],
                "prediction": prediction[i],
                "probabilities": probabilities[i],
            }
        )




main()

