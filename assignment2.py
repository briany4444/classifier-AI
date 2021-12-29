import math

# Brian Yang 101140298 COMP3106 A2


# these functions will be used for both classifiers

# finds the image proportions for black pixels
# and the proportion of black pixels in top and left half
# returns them as an array (aka the feature vector)
def findImageProportions(grid):
    evidence = []
    totalBlackPixels = findTotalBlack(grid)
    evidence.append(findPropBlack(grid, totalBlackPixels))
    evidence.append(findTopProp(grid, totalBlackPixels))
    evidence.append(findLeftProp(grid, totalBlackPixels))
    return evidence

# finds the number of black pixels in the image, given a grid
def findTotalBlack(grid):
    totalBlackPixels = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "1":
                totalBlackPixels = totalBlackPixels + 1
    return totalBlackPixels

# finds and returns proportion of pixels in grid that are black ("1")
def findPropBlack(grid, totalBlackPixels):
    totalPixels = len(grid) * len(grid[0])
    propBlack = 0
    if totalPixels > 0:
        propBlack = totalBlackPixels / totalPixels
    return propBlack

# finds and returns the proportion of black pixels that are in the top half
def findTopProp(grid, totalBlackPixels):
    topBlackPixels = 0

    # checks the top half of grid
    # finding the int value, I assume that the middle row is NOT in the top half if there is an odd number
    middleRowIndex = math.floor(len(grid) / 2)

    for i in range(middleRowIndex):
        for j in range(len(grid[i])):
            if grid[i][j] == "1":
                 topBlackPixels = topBlackPixels + 1
    topProp = 0
    if totalBlackPixels > 0:
        topProp = topBlackPixels / totalBlackPixels

    return topProp

# finds and returns the proportion of black pixels that are on the left side
def findLeftProp(grid, totalBlackPixels):
    leftBlackPixels = 0

    # NOTE: I assume middle row is NOT on the left half if there is an odd number of columnms
    # only checks the left half of each row
    middleColIndex = math.floor(len(grid[0]) / 2)

    for i in range(len(grid)):
        for j in range(middleColIndex):
            if grid[i][j] == "1":
                 leftBlackPixels = leftBlackPixels + 1
    leftProp = 0
    # makes sure theres no 0 division
    if totalBlackPixels > 0:
        leftProp = leftBlackPixels / totalBlackPixels

    return leftProp

# creates and returns the image as a 2D list of characters from a csv file
def createGrid(input_filepath):
    grid = []
    file = open(input_filepath, "r")
    for l in file:
        line = []
        l = l.strip()
        for pixel in l:
            if pixel == ",":
                continue
            line.append(pixel)
        grid.append(line)
    file.close()
    return grid


# -----------------------------------------------
#             NAIVE BAYES CLASSIFIER
# -----------------------------------------------

# applies the naive bayes assumption calculation given a particular class
# ex. P(e1 and e2 and e3|class) = P(e1|class) * P(e2|class) * P(e3|class)
# returns the calculated value by using the conditional probability of a feature calculation
def naiveBayesAssumption(classNum, evidence, bayesKnowns):
    probability = 1
    for evidenceIndex in range(len(evidence)):
        currEvidence = evidence[evidenceIndex]
        probability = probability * evidenceGivenClass(evidenceIndex, classNum, currEvidence, bayesKnowns)
    return probability

# calculates the probability of the evidence intersects (the denominator) and returns the calculation
# by using the naive bayes assumption for every class
# ex. does P(e1 and e2 and e3)
def probEvidenceIntersect(priorProb, evidence, bayesKnowns):
    probability = 0
    for classIndex in range(len(priorProb)):
        probability = probability + naiveBayesAssumption(classIndex, evidence, bayesKnowns)*priorProb[classIndex]
    return probability

# calculates P(feature|class) using knowns and the piece of evidence
# currEvidence is "x" in the formula provided (the feature proportion)
def evidenceGivenClass(evidenceIndex, classIndex, currEvidence, bayesKnowns):
    # based on hardcoded mu and sigma values from pdf for each class and evidence
    mu = bayesKnowns[classIndex][evidenceIndex][0]
    sigma = bayesKnowns[classIndex][evidenceIndex][1]

    # uses the normal distribution calculation provided from pdf to find probability
    # of a piece of evidence given a class hypothesis
    root = math.sqrt(2*(math.pi)*(sigma**2))
    exponent = math.exp(-(1/2)*(((currEvidence - mu) / sigma)**2))
    probability = exponent / root

    return probability

# input is the full file path to a CSV file containing a matrix representation of a black-and-white image

# most_likely_class is a string indicating the most likely class, either "A", "B", "C", "D", or "E"
# class_probabilities is a five element list indicating the probability of each class in the order [A probability, B probability, C probability, D probability, E probability]
def naive_bayes_classifier(input_filepath):
    class_probabilities = []
    classes = ["A", "B", "C", "D", "E"]

    # the prior probabilities for each class in alphabetical order
    priorProb = [0.28, 0.05, 0.10, 0.15, 0.42]

    # 3D list filled with values from pdf
    # first dimension is class
    # second dimension is [propBlack, topProp, leftProp] for a class
    # third dimension is [mu, sigma]
    bayesKnowns = [[[0.38, 0.06], [0.46, 0.12], [0.50, 0.09]],      # A
                   [[0.51, 0.06], [0.49, 0.12], [0.57, 0.09]],      # B
                   [[0.31, 0.06], [0.37, 0.09], [0.64, 0.06]],      # C
                   [[0.39, 0.06], [0.47, 0.09], [0.57, 0.03]],      # D
                   [[0.43, 0.12], [0.45, 0.15], [0.65, 0.09]]]      # E

    # creates grid and gets feature vector from the grid
    grid = createGrid(input_filepath)
    evidence = findImageProportions(grid)

    # this denominator is used for all the bayes calculations
    denominator = probEvidenceIntersect(priorProb, evidence, bayesKnowns)

    # calculates posterior for each class
    for classIndex in range(len(classes)):
        # calculates the numerator for each hypothesis in bayes theorom
        numerator = naiveBayesAssumption(classIndex, evidence, bayesKnowns) * priorProb[classIndex]
        class_probabilities.append(numerator / denominator)

    # assuming no ties in class probabilities
    highestProbability = max(class_probabilities)
    index = class_probabilities.index(highestProbability)
    most_likely_class = classes[index]

    # note that the highest probability will provide us with the same answer as the naive bayes classifier (same thing)

    return most_likely_class, class_probabilities


# -------------------------------------------------------
#                   FUZZY CLASSIFIER
# -------------------------------------------------------

# this calculates an output for a trapezoidal membership function
# given an input x (in this case it is a piece of evidence)
def trapezoidMembership(x, characteristics):
    a = characteristics[0]
    b = characteristics[1]
    c = characteristics[2]
    d = characteristics[3]
    # if outside of support, by default returns 0
    output = 0
    if a < x and x < b:
        output = (x-a) / (b-a)
    elif b <= x and x <= c:
        output = 1
    elif c < x and x < d:
        output = (d-x) / (d-c)
    return output

# for the following rules
# note that class_membership_func(x): {1 if x is [insert class]    0 otherwise} for consequence
# returns output membership function for each rule by finding the antecendent
def membershipA(characteristics, evidence):
    # constants
    PROPBLACK = 0
    TOPPROP = 1
    LEFTPROP = 2
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    propBlackTrap = characteristics[PROPBLACK][MEDIUM]
    topPropTrap = characteristics[TOPPROP][MEDIUM]
    leftPropTrap = characteristics[LEFTPROP][MEDIUM]

    # gets truth value of each premise
    propBlackOut = trapezoidMembership(evidence[PROPBLACK], propBlackTrap)
    topPropOut = trapezoidMembership(evidence[TOPPROP], topPropTrap)
    leftPropOut = trapezoidMembership(evidence[LEFTPROP], leftPropTrap)

    # gets rule strength
    ruleStr = min(propBlackOut, max(topPropOut, leftPropOut))

    # clipped with rulestr
    output = min(ruleStr, 1)

    return output

def membershipB(characteristics, evidence):
    PROPBLACK = 0
    TOPPROP = 1
    LEFTPROP = 2
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    propBlackTrap = characteristics[PROPBLACK][HIGH]
    topPropTrap = characteristics[TOPPROP][MEDIUM]
    leftPropTrap = characteristics[LEFTPROP][MEDIUM]

    propBlackOut = trapezoidMembership(evidence[PROPBLACK], propBlackTrap)
    topPropOut = trapezoidMembership(evidence[TOPPROP], topPropTrap)
    leftPropOut = trapezoidMembership(evidence[LEFTPROP], leftPropTrap)

    ruleStr = min(min(propBlackOut, topPropOut), leftPropOut)

    output = min(ruleStr, 1)

    return output

def membershipC(characteristics, evidence):
    PROPBLACK = 0
    TOPPROP = 1
    LEFTPROP = 2
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    propBlackTrap = characteristics[PROPBLACK][LOW]
    topPropTrap = characteristics[TOPPROP][MEDIUM]
    leftPropTrap = characteristics[LEFTPROP][HIGH]

    propBlackOut = trapezoidMembership(evidence[PROPBLACK], propBlackTrap)
    topPropOut = trapezoidMembership(evidence[TOPPROP], topPropTrap)
    leftPropOut = trapezoidMembership(evidence[LEFTPROP], leftPropTrap)

    ruleStr = max(min(propBlackOut, topPropOut), leftPropOut)

    output = min(ruleStr, 1)

    return output

def membershipD(characteristics, evidence):
    PROPBLACK = 0
    TOPPROP = 1
    LEFTPROP = 2
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    propBlackTrap = characteristics[PROPBLACK][MEDIUM]
    topPropTrap = characteristics[TOPPROP][MEDIUM]
    leftPropTrap = characteristics[LEFTPROP][HIGH]

    propBlackOut = trapezoidMembership(evidence[PROPBLACK], propBlackTrap)
    topPropOut = trapezoidMembership(evidence[TOPPROP], topPropTrap)
    leftPropOut = trapezoidMembership(evidence[LEFTPROP], leftPropTrap)

    ruleStr = min(min(propBlackOut, topPropOut), leftPropOut)

    output = min(ruleStr, 1)

    return output

def membershipE(characteristics, evidence):
    PROPBLACK = 0
    TOPPROP = 1
    LEFTPROP = 2
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    propBlackTrap = characteristics[PROPBLACK][HIGH]
    topPropTrap = characteristics[TOPPROP][MEDIUM]
    leftPropTrap = characteristics[LEFTPROP][HIGH]

    propBlackOut = trapezoidMembership(evidence[PROPBLACK], propBlackTrap)
    topPropOut = trapezoidMembership(evidence[TOPPROP], topPropTrap)
    leftPropOut = trapezoidMembership(evidence[LEFTPROP], leftPropTrap)

    ruleStr = min(min(propBlackOut, topPropOut), leftPropOut)

    output = min(ruleStr, 1)

    return output


# input is the full file path to a CSV file containing a matrix representation of a black-and-white image

# highest_membership_class is a string indicating the highest membership class, either "A", "B", "C", "D", or "E"
# class_memberships is a four element list indicating the membership in each class in the order [A value, B value, C value, D value, E value]
def fuzzy_classifier(input_filepath):
    class_memberships = []
    classes = ["A", "B", "C", "D", "E"]

    # hardcoded from pdf
    # first dimension is the feature type
    # second dimension is the subtype of feature (low, medium, high)
    # third dimension is the abcd for a trapezoid
    characteristics = [[[0, 0, 0.3, 0.4], [0.3, 0.4, 0.4, 0.5], [0.4, 0.5, 1, 1]],   # low
                       [[0, 0, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6], [0.5, 0.6, 1, 1]],   # medium
                       [[0, 0, 0.3, 0.4], [0.3, 0.4, 0.6, 0.7], [0.6, 0.7, 1, 1]]]   # high
    # creates grid and image proportions
    grid = createGrid(input_filepath)
    evidence = findImageProportions(grid)

    # gets all the membership function values
    # there is a different function for each rule because they all
    # use different premises and connectives
    class_memberships.append(membershipA(characteristics, evidence))
    class_memberships.append(membershipB(characteristics, evidence))
    class_memberships.append(membershipC(characteristics, evidence))
    class_memberships.append(membershipD(characteristics, evidence))
    class_memberships.append(membershipE(characteristics, evidence))

    # assuming the highest probability is unique
    highestProbability = max(class_memberships)
    index = class_memberships.index(highestProbability)
    highest_membership_class = classes[index]

    return highest_membership_class, class_memberships

print(naive_bayes_classifier("Examples/Example4/input.csv"))
