# Définition du format de sortie
set terminal pngcairo

# Définition du nom du fichier de sortie
set output 'courbeEgalisation.png'

# Définition des titres de la figure
set title 'CPU vs. GPU - Egalisation histogramme' # titre
set xlabel 'Taille image (Mega Pixels)'     # nom de l'axe des abscisses
set ylabel 'Temps de calcul (ms)'            # nom de l'axe des ordonnées
set y2label 'Speedup CPU/GPU'        # nom de l'axe des ordonnées
set y2tics nomirror

plot 'outputEgalisationCPU.data' axis x1y1 with lines title "Temps calcul CPU", 'outputEgalisationGPU.data' axis x1y1 with lines title "Temps calcul GPU", 'outputEgalisationRatio.data' axis x1y2 with lines title "Speedup CPU/GPU"
#plot 'outputEgalisationGPU.data' axis x1y1 with lines title "Temps calcul GPU", 'outputEgalisationRatio.data' axis x1y2 with lines title "Speedup CPU/GPU"
