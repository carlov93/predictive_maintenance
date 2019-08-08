## Theorie 
Die Wavelet-Transformation kann als Verbesserung der Kurzzeit-Fourier-Transformation(STFT) angesehen werden. <br>

Die Kurzzeit-Fourier-Transformation (englisch short-time Fourier transform, kurz STFT) ist eine Methode aus der Fourier-Analysis, um die zeitliche Änderung des Frequenzspektrums eines Signals darzustellen. Während die Fourier-Transformation keine Informationen über die zeitliche Veränderung des Spektrums bereitstellt, ist die STFT auch für nichtstationäre Signale geeignet, deren Frequenzeigenschaften sich im Laufe der Zeit verändern.  <br>

Zur Transformation wird das Zeitsignal in einzelnen Zeitabschnitte mit Hilfe einer Fensterfunktion unterteilt und diese einzelnen Zeitabschnitte in jeweils einzelne Spektralbereiche überführt. Die zeitliche Aneinanderreihung der so gewonnenen Spektralbereiche stellt die STFT dar, welche sich dreidimensional oder in Flächendarstellung mit verschiedenen Farben grafisch darstellen lässt.  <br>

Python Library:  <br>
https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html