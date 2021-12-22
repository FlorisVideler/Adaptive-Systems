# Double Deep Q-learning Networks in the Gym

Floris Videler:		1758374 <br>
Niels Bijl:		1754339

<b>Gedrag agent</b><br>
De agent leert met de implementatie waarin wij een klein beetje zijn afgeweken van de pseudocode op canvas, zo goed als perfect. Bij episode 200-300 behaald de agent vaak voor het eerst een reward van +- 200 en landt dus al goed. De gemiddelde reward tussen episode 300 en 400 is 200+. Daarmee is te stellen dat de agent het bij episode 400 wel degelijk goed geleerd heeft om te landen. Als we door trainen en wachten tot het totale gemiddelde op de 200 zit is dit vaak tussen de 1500 en 2000 episodes
<br>

<b>Afwijking pseudocode canvas</b> <br>
Wanneer wij 1:1 de pseudocode van canvas overnemen is onze agent niet in staat om zichzelf aan te leren om succesvol te landen. Wij denken dat er iets mis gaat met de pseudo code en zijn daarom iets afgeweken van canvas. Het vaststellen van de beste actie met het policy netwerk lijkt niet juist te zijn. Door het voorspellen van de beste actie met het target netwerk en met die uitkomst de target te berekenen en het policy netwerk te trainen behaalde wij al snel beter resultaat.
<br>

<b>Tijd</b> <br>
Tensorflow v2 is extreem traag met onze lunar lander implementatie. Dit komt door het gebruik van “eager execution”. In Tensorflow v1 is het mogelijk om eager execution uit te zetten dit kan het trainen tot wel 21 keer zo snel maken. Waardoor we een stuk sneller resultaat zagen.
<br>

![Lunar landing learning_curv](https://i.ibb.co/NSp9Krr/lunarlanding.png)

<br>

![Lunar landing gif](https://github.com/florisvideler/Adaptive-Systems/blob/main/double-deep-q-learning-networks-in-the-gym/assets/lunar.gif?raw=true)