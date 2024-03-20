import cv2
import os
import numpy as np
import supervised_learning
from skimage.metrics import structural_similarity as ssim


def get_countours_of_img(img_name, test_folder):
    img = cv2.cvtColor(cv2.imread(os.path.join(test_folder, img_name)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY), (100, 100))
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    ret1, threshold_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
    invert_img = 255 - morph_img
    contours, ret2 = cv2.findContours(invert_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_img = cv2.drawContours(np.zeros(invert_img.shape, np.uint8), contours, -1, 255, 1)
    return contours[0]


def contour_comparison_between_two(inp, test_folder):
    img1, img2 = inp.split(" ")
    img1_contours = get_countours_of_img(img1, test_folder)
    img2_contours = get_countours_of_img(img2, test_folder)
    similarity = cv2.matchShapes(img1_contours, img2_contours, 1, 0)
    print(f"Slichnosta pomegju ovie dva potpisi presmetana spored nejzinite konturi e: {similarity}\n"
          f"[Info: Pomali brojki bliski do 0 znachat deka potpisite se rechisi identichni, \n"
          f"a pogolemi brojki znachat deka potpisite se razlichni. Imajte na um deka ovie vrednosti\n"
          f"ne se celosno verodostojni, odnosno metodite za nivno dobivanje imaat odredeni nedostatoci.]")


def SSIM_comparison_between_two(inp, test_folder):
    img1, img2 = inp.split(" ")
    img1 = cv2.cvtColor(cv2.imread(os.path.join(test_folder, img1)), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(cv2.cvtColor(img1.copy(), cv2.COLOR_RGB2GRAY), (100, 100))
    img2 = cv2.cvtColor(cv2.imread(os.path.join(test_folder, img2)), cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(cv2.cvtColor(img2.copy(), cv2.COLOR_RGB2GRAY), (100, 100))
    similarity = ssim(img1, img2)
    print(f"Slichnosta pomegju ovie dva potpisi presmetana spored SSIM e: {similarity}\n"
          f"[Info: Vrednostite se dvizhat od 0 do 1, odnosno dokolku vrednosta e blisku do 0\n"
          f"znachi deka potpisite ne se slichni, dodeka dokolku e blisku do 1, znachi deka\n"
          f"potpisite se mnogu slichni. Sepak, zaradi samata priroda na sporedbata, vrednosti\n"
          f"okolu 0.6 ili povisoko kaj Avtor1 i Avtor2 znachat deka potpisite se mnogu slichni, \n"
          f"dodeka kaj Avtor3 vrednostite na slichnite potpisi se dvizhat okolu 0.35 i nagore. \n"
          f"Imajte na um deka ovie vrednosti ne se celosno verodostojni, odnosno metodite za nivno \n"
          f"dobivanje imaat odredeni nedostatoci.]")


def similarity_between_two(test_folder):
    inp = input("Vnesete dve sliki oddeleni so prazno mesto od test folderot\n")
    contoursOrSSIM = input("Izberete nachin na koj sakate da ja najdete slichnosta\n"
                           "[Vnesete reden broj]\n"
                           "1. Slichnost pomegju konturite na potpisite\n"
                           "2. Slichnost so pomosh na SSIM\n")
    if contoursOrSSIM == "1":
        contour_comparison_between_two(inp, test_folder)
    elif contoursOrSSIM == "2":
        SSIM_comparison_between_two(inp, test_folder)
    else:
        print("Vnesete validen broj!")
        return


def contour_comparison_with_all_other(inp, test_folder):
    query_image = get_countours_of_img(inp.split(" ")[0], test_folder)
    similarities = {}
    images = []
    for file in os.listdir(test_folder):
        if os.path.isfile(os.path.join(test_folder, file)):
            images.append(os.path.join(test_folder, file))
    for img in images:
        contours = get_countours_of_img(img, test_folder)
        similarities[os.path.basename(img)] = cv2.matchShapes(query_image, contours, 1, 0)
    for k, v in sorted(similarities.items(), key=lambda x: x[1]):
        print('%9s:\t%-25s' % (k, v))
    print(f"Slichnosta pomegju ovoj potpis i ostanatite presmetana spored konturi e dadena nad ovoj tekst.\n"
          f"[Info: Pomali brojki bliski do 0 znachat deka potpisite se rechisi identichni, a pogolemi \n"
          f"brojki znachat deka potpisite se razlichni. Redosledot na pechatenje na slichnostite e od \n"
          f"najgolema kon najmala. Imajte na um deka ovie vrednosti ne se celosno verodostojni, odnosno\n"
          f"metodite za nivno dobivanje imaat odredeni nedostatoci.]")


def SSIM_comparison_with_all_other(inp, test_folder):
    img = cv2.cvtColor(cv2.imread(os.path.join(test_folder, inp)), cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY), (100, 100))
    images = []
    for file in os.listdir(test_folder):
        if os.path.isfile(os.path.join(test_folder, file)):
            images.append(os.path.join(test_folder, file))
    similarities = {}
    for path in images:
        img = cv2.cvtColor(cv2.imread(os.path.join(test_folder, path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY), (100, 100))
        if path != inp:
            similarities[path.split("/")[-1]] = ssim(original_image, img)
    maxx = max(similarities.items(), key=lambda x: x[1])
    similarities.pop(maxx[0])
    maxx = max(similarities.items(), key=lambda x: x[1])[1]
    for k, v in similarities.items():
        similarities[k] = similarities[k] / maxx * 100
    for k, v in sorted(similarities.items(), key=lambda x: x[1]):
        print('%25s:\t%.2f%%' % (k, v))
    print(f"Slichnosta pomegju ovoj potpis i ostanatite presmetana spored SSIM e dadena nad ovoj tekst.\n"
          f"[Info: Vrednostite se dadeni vo procenti, odnosno vrednosti bliski do 0 procenti znachat\n"
          f"deka potpisot ne e slichen so vneseniot, dodeka pak vrednosti bliski do 100 znachat\n"
          f"deka potpisot e mnogu slichen so vneseniot. Imajte na um deka ovie vrednosti ne se celosno\n"
          f"verodostojni, odnosno metodite za nivno dobivanje imaat odredeni nedostatoci.]")


def similarity_between_one_and_all_other(test_folder):
    inp = input("Vnesete slika: \n")
    contoursOrSSIM = input("Izberete nachin na koj sakate da ja najdete slichnosta\n"
                           "[Vnesete reden broj]\n"
                           "1. Slichnost pomegju konturite na potpisite\n"
                           "2. Slichnost so pomosh na SSIM\n")
    if contoursOrSSIM == "1":
        contour_comparison_with_all_other(inp, test_folder)
    elif contoursOrSSIM == "2":
        SSIM_comparison_with_all_other(inp, test_folder)
    else:
        print("Vnesete validen broj!")
        return
