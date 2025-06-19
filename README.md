# Arta Codului Genetic: Un Generator de Arta ASCII

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Functional-brightgreen?style=for-the-badge)

Un proiect care utilizeaza puterea algoritmilor genetici pentru a transforma imagini digitale in arta compusa din caractere text (ASCII Art), cautand cea mai buna reprezentare vizuala printr-un proces evolutiv.

## Demonstratie Vizuala

| Imagine Originala | Arta ASCII Generata |
| :---------------: | :-----------------: |
|  *pune aici o imagine originala* | *pune aici o imagine cu arta ASCII generata* |
| `input/mr_bean.jpg` | `output/mr_bean_ascii.txt` |

## Caracteristici Principale

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

## Utilizare

Scriptul se ruleaza din linia de comanda, specificand imaginea de intrare si, optional, alti parametri.

#### Exemplu Simplu
```bash
python main.py input/mr_bean.jpg
