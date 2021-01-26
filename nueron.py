#!/usr/bin/python3
import math


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def activate(inputs, weights):
    #perform net input
    sum = 0
    for x, w in zip(inputs, weights):
        sum += x * w

    #perfrom activation
    return sigmoid(sum)


def main():
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print(output)


if __name__ == "__main__":
    main()

