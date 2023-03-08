# Définition du format de sortie
set terminal pngcairo

# Définition du nom du fichier de sortie
set output 'courbeEtirement.png'

# Définition des titres de la figure
set title 'CPU vs. GPU - Etirement histogramme' # titre
set xlabel 'Taille image (Mega Pixels)'     # nom de l'axe des abscisses
set ylabel 'Temps de calcul (ms)'            # nom de l'axe des ordonnées
set y2label 'Speedup CPU/GPU'        # nom de l'axe des ordonnées
set y2tics nomirror

plot 'outputEtirementCPU.data' axis x1y1 with lines title "Temps calcul CPU", 'outputEtirementGPU.data' axis x1y1 with lines title "Temps calcul GPU", 'outputEtirementRatio.data' axis x1y2 with lines title "Speedup CPU/GPU"
#plot 'outputEtirementGPU.data' axis x1y1 with lines title "Temps calcul GPU", 'outputEtirementRatio.data' axis x1y2 with lines title "Speedup CPU/GPU"
