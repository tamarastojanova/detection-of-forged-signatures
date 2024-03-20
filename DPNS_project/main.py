import os
import similarities
import supervised_learning

if __name__ == '__main__':
    author = input("Izberete avtor na potpisi pomegju avtorite: Avtor1, Avtor2 ili Avtor3\n"
                   "[Vnesete vo format: 'Avtor#']\n")
    whatToDo = input("Izberete shto sakate da pravite preku vnesuvanje na redniot broj.\n"
                     "1. Proverka dali eden potpis e verodostoen vrz osnova na Supervised Learning.\n"
                     "2. Naogjanje na slichnost pomegju dva potpisi.\n"
                     "3. Naogjanje na slichnost na potpis so ostanatite potpisi.\n")
    current_dir = os.path.dirname(__file__)
    test_folder = os.path.join(current_dir, 'data/test/', author)
    if whatToDo == "1":
        supervised_learning.learn(author, current_dir)
    elif whatToDo == "2":
        similarities.similarity_between_two(test_folder)
    elif whatToDo == "3":
        similarities.similarity_between_one_and_all_other(test_folder)
    else:
        print("Vnesete validen broj!")
        exit(1)
