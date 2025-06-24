# ASCII Art Genetic

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Functional-brightgreen?style=for-the-badge)

Un proiect care utilizeaza puterea algoritmilor genetici pentru a transforma imagini digitale in arta compusa din caractere text (ASCII Art), cautand cea mai buna reprezentare vizuala printr-un proces evolutiv.

## Functionalitatea Codului

| Imagine Originala | Arta ASCII Generata |
| :---------------: | :-----------------: |
| <img src="https://github.com/user-attachments/assets/d004beb0-ce13-4803-84a2-02976a7ae242" width="300"/> | <img src="https://github.com/user-attachments/assets/1e5d51b7-2ee9-43b7-98fe-04d780633534" width="300"/> |

| Evolutia Fitness-ului Genetic |
| :-----------------: |
| ![image](https://github.com/user-attachments/assets/73cb07d3-fc5b-4a32-98e4-6952e91ece5a) |




## Caracteristici 

-   **Conversie Imagine-ASCII**: Transforma orice imagine (`.jpg`, `.png`, etc.) intr-o matrice de caractere.
-   **Optimizare cu Algoritmi Genetici**: Nu alege pur si simplu caractere; evolueaza o populatie de solutii pentru a gasi combinatia optima care imita cel mai bine imaginea sursa.
-   **Functie de Fitness Bazata pe SSIM**: Asemanarea este masurata folosind *Structural Similarity Index (SSIM)*, o metrica avansata care pastreaza structura imaginii.
-   **Analiza Performantei**: Genereaza grafice detaliate care arata evolutia performantei (fitness-ului) de-a lungul generatiilor.
-   **Interfata Linie de Comanda**: Parametri precum dimensiunea populatiei, rata de mutatie si altii pot fi controlati direct din terminal pentru flexibilitate maxima.

## Tehnologii Utilizate

-   **Python**: Limbajul de baza al proiectului.
-   **Pillow (PIL Fork)**: Pentru toate operatiunile de manipulare a imaginilor (incarcare, redimensionare, conversie grayscale).
-   **NumPy**: Pentru operatii numerice rapide si eficiente pe matricile de pixeli.
-   **scikit-image**: Pentru calcularea metricii SSIM, nucleul functiei de fitness.
-   **Matplotlib**: Pentru vizualizarea si salvarea graficelor de performanta.
-   **Argparse**: Pentru a crea o interfata de linia de comanda (CLI) robusta si usor de folosit.

## Resurse

|        Pillow     |          MatLab     |
| :---------------: | :-----------------: |
| https://pillow.readthedocs.io/en/stable/ | https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/images/events/matlabexpo/kr/2021/how-to-use-matlab-with-python.pdf |



## Utilizare

Scriptul se ruleaza din linia de comanda, specificand imaginea de intrare si, optional, alti parametri.

#### Exemplu Simplu
```bash
python main.py input/mr_bean.jpg
