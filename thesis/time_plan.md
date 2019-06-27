# Grober Zeitplan
## 1. Phase (Ende September)
- LSTM zum Laufen bringen und gucken ob die Zeitreihe rekontruiert werden kann
    - Ab einem gewissen Zeitpunkt in der Zeitreihe einen random walk hinzufügen (diesen darf das LSTM nicht lernen können)
    - LSTM soll im Grunde genommen nicht in die Falle der Autocorellation und des Random Walkes verfallen
    - zu zeigen ist, dass das Netz dies beides nicht lernt

## 2. Phase (Ende Oktober)
- Die Aufgabe eines neuronalen Netzes ist es etwas Unbekanntes auf etwas Bekanntes (Trainingsdaten) zurück zuführen
- Dafür können verschiedene Maße verwendet werden 
- Implement reconstruction (MSE) and maximum likelihood (Energiesatz) metric and analyse it

## 3. Phase (Ende Dezember)
- LSTM memory cell states as a thrid metric usefull?
   - Wie stark ändern sich die LSTM Zellen zwischen dem Maschinenzustand welcher ok ist und dem Zustand der nicht ok ist?
   - Eine starke Änderung zeigt, dass etwas ungewöhnliches passiert ist

## 4. Phase (Abgabe der Masterarbeit Ende Februar)
- Paper parallel zur Programmierung anfertigen 
- Paper auf Englsich, ca. 50-60 Seiten, fertige! Masterarbeitsteile können während der 6 monatigen Bearbeitungszeit dem Prof. geschickt werden um feedback zu erhalten