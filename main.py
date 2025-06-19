# main.py
import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# --- UTILS (Funcții ajutătoare) ---

def load_and_prepare_image(image_path, target_width=120):
    """Încarcă o imagine, o convertește în grayscale și o redimensionează."""
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Eroare: Fișierul '{image_path}' nu a fost găsit.")
        return None
    
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio * 0.5)
    img = img.resize((target_width, target_height))
    
    return np.array(img) / 255.0

def get_character_intensities(char_set):
    """Calculează 'luminozitatea' fiecărui caracter."""
    intensities = {}
    try:
        font = ImageFont.truetype("cour.ttf", 15) # Courier New e o alegere bună
    except IOError:
        font = ImageFont.load_default()
        
    for char in char_set:
        char_img = Image.new('L', (10, 20), color=255)
        draw = ImageDraw.Draw(char_img)
        draw.text((0, 0), char, fill=0, font=font)
        intensity = np.array(char_img).mean() / 255.0
        intensities[char] = intensity
    return intensities

def plot_fitness_evolution(fitness_history, output_path):
    if not fitness_history:
        return
    generations = list(range(1, len(fitness_history)+1))
    plt.figure(figsize=(10, 5))
    plt.plot(generations, fitness_history, label='Best Fitness', color='green')
    plt.xlabel('Generatie')
    plt.ylabel('Fitness (SSIM)')
    plt.title('Evolutia Fitness-ului Algoritmului Genetic')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Grafic fitness salvat in: {output_path}")

def chromosome_to_text(chromosome, char_set):
    return "\n".join("".join(char_set[index] for index in row) for row in chromosome)

def classic_ascii_art(image_array, char_set):
    # Normalizează imaginea la [0, 1]
    norm_img = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    char_list = list(char_set)
    n_chars = len(char_list)
    result = []
    for row in norm_img:
        line = ''.join(char_list[int(val * (n_chars - 1))] for val in row)
        result.append(line)
    return '\n'.join(result)

# --- CLASA PENTRU ALGORITMUL GENETIC ---

class GeneticAsciiArt:
    def __init__(self, target_image_array, char_set, population_size=50, mutation_rate=0.01):
        self.target_image = target_image_array
        self.height, self.width = target_image_array.shape
        self.char_set = list(char_set)
        
        self.char_intensities_map = get_character_intensities(self.char_set)
        self.intensity_values = np.array([self.char_intensities_map[c] for c in self.char_set])
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self):
        return [np.random.randint(0, len(self.char_set), size=(self.height, self.width)) for _ in range(self.population_size)]

    def _chromosome_to_image(self, chromosome):
        return self.intensity_values[chromosome]

    def _calculate_fitness(self, chromosome):
        candidate_image = self._chromosome_to_image(chromosome)
        return ssim(self.target_image, candidate_image, data_range=1.0)

    def _selection(self, fitness_scores, k=3):
        selection_ix = random.randrange(len(self.population))
        for _ in range(k - 1):
            ix = random.randrange(len(self.population))
            if fitness_scores[ix] > fitness_scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]

    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.height * self.width - 2)
        flat_p1 = parent1.flatten()
        flat_p2 = parent2.flatten()
        flat_p1[crossover_point:] = flat_p2[crossover_point:]
        return flat_p1.reshape((self.height, self.width))

    def _mutation(self, chromosome):
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.mutation_rate:
                    chromosome[i, j] = random.randint(0, len(self.char_set)-1)
        return chromosome

    def run(self, generations):
        for gen in tqdm(range(generations), desc="Evoluție", ncols=100):
            fitness_scores = [self._calculate_fitness(chromo) for chromo in self.population]
            self.fitness_history.append(max(fitness_scores))
            
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._selection(fitness_scores)
                parent2 = self._selection(fitness_scores)
                child = self._crossover(parent1, parent2)
                mutated_child = self._mutation(child)
                new_population.append(mutated_child)
            
            self.population = new_population
        
        final_fitness_scores = [self._calculate_fitness(chromo) for chromo in self.population]
        best_index = np.argmax(final_fitness_scores)
        return self.population[best_index]

# --- ARGUMENTE CLI ---

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASCII Art Generator cu Algoritm Genetic')
    parser.add_argument('image', help='Calea catre imaginea de procesat (input sau output folder)')
    parser.add_argument('--width', type=int, default=150, help='Latimea in caractere (default: 150)')
    parser.add_argument('--population', type=int, default=200, help='Dimensiunea populatiei (default: 200)')
    parser.add_argument('--generations', type=int, default=400, help='Numarul de generatii (default: 400)')
    parser.add_argument('--mutation', type=float, default=0.003, help='Rata de mutatie (default: 0.003)')
    parser.add_argument('--charset', type=str, default=" .:-=+*#%@",
                        help='Setul de caractere (default: de la deschis la inchis)')
    parser.add_argument('--classic', action='store_true', help='Genereaza rapid ASCII art clasic (fara genetic)')
    return parser.parse_args()

# --- MAIN ---

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Utilizare: python main.py <imagine> [--width ...] [--population ...] [--generations ...] [--mutation ...] [--charset ...] [--classic]")
        print("Exemplu: python main.py test3.jpg --classic")
        sys.exit(1)

    args = parse_arguments()
    img_path = args.image
    if not os.path.isfile(img_path):
        for folder in ["input", "output"]:
            alt_path = os.path.join(folder, args.image)
            if os.path.isfile(alt_path):
                img_path = alt_path
                break
    if not os.path.isfile(img_path):
        print(f"Eroare: Fișierul '{args.image}' nu a fost găsit în folderul curent, input/ sau output/")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    OUTPUT_TXT_PATH = f"output/{base_name}.txt"
    OUTPUT_GRAPH_PATH = f"output/{base_name}_fitness.png"
    OUTPUT_IMG_PATH = f"output/{base_name}.png"
    os.makedirs("output", exist_ok=True)

    print(f"Se procesează imaginea: {img_path}")
    target_image = load_and_prepare_image(img_path, target_width=args.width)
    if target_image is None:
        sys.exit(1)

    if args.classic:
        final_ascii_art = classic_ascii_art(target_image, args.charset)
        print("ASCII art generat rapid (fără genetic).")
        # Nu există fitness/graph pentru varianta clasică
    else:
        ga = GeneticAsciiArt(target_image, args.charset, args.population, args.mutation)
        best_chromosome = ga.run(args.generations)
        final_ascii_art = chromosome_to_text(best_chromosome, list(args.charset))
        plot_fitness_evolution(ga.fitness_history, OUTPUT_GRAPH_PATH)

    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write(final_ascii_art)
    print(f"\nArtă ASCII salvată în '{OUTPUT_TXT_PATH}'")

    # Salvare imagine ASCII ca PNG
    try:
        font = ImageFont.truetype("cour.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    lines = final_ascii_art.split('\n')
    lines = [line for line in lines if line]
    char_width, char_height = font.getsize('A')
    img_width = char_width * len(lines[0]) if lines else 0
    img_height = char_height * len(lines)
    img = Image.new('L', (img_width, img_height), color=255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((0, i * char_height), line, fill=0, font=font)
    img.save(OUTPUT_IMG_PATH)
    print(f"Imaginea ASCII salvată în '{OUTPUT_IMG_PATH}'")

# Pentru a genera ASCII art pentru mr.bean.jpg:
# Rulează în terminal:
# python main.py mr.bean.jpg --classic
# sau cu parametri suplimentari:
# python main.py mr.bean.jpg --width 150 --classic
# Poți folosi și varianta genetică fără --classic dacă vrei.