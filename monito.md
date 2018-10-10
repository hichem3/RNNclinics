# RNNclinics
Il y a trois algorithmes possibles pour créer des représentations des mots (word embeddings) dans un réseau de neurones: Embedding Layer, Word2Vec, GlOve.

----------------------------------------------------------------------------------------------------------------------------------
Embedding Layer.
Dans le code "RNN+LSTM" on voit la façon simple, j'ai crée un Embedding Layer qui est en fait une couche qui est entrainé directement dedans le réseau des neurones.

----------------------------------------------------------------------------------------------------------------------------------
Word2Vec.
Ici il y a deux architectures possibles: Skip-Gram et Bag-of-Words (https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures) 
Dans le code de "Word2Vec model on IMDB reviews"  j’ai créé moi-même un modelé word2vec qui utilise la méthode Skip-Gram à partir des textes des reviews. Je n'ai pas encore testé le modèle ainsi obtenu sur le problème de prédiction des sentiments.
J'ai aussi écrit un code qui utilise la méthode Bag-of-words et entraine le modèle avec une forêt aléatoire (code pas publié).

En outre  j'ai mis le code "RNN+LSTM+word2vec" où il y a un algorithme qui entraine une LSTM (les détails modèle du réseau que j'ai utilisé ici ne sont pas importants) en utilisant un word embedding issu d’une très grande base de données (plus que 3 milliards de mots) pre-entrainé par Google. Néanmoins, les résultats obtenus après l’entrainement du modèle sont très médiocres... Ceci est aussi cause du fait que la LSTM a seulement 16 unités, les epochs sont seulement 6, les séquences sont coupées à 30 mots. Tout cela peut être amélioré. 

----------------------------------------------------------------------------------------------------------------------------------
GlOve.
Comme pour le cas précèdentl dans le code "RNN+LSTM+GlOve" il y a un algorithme qui entraine une LSTM à partir d'un word embedding issu d’une très grande base de données pre-entrainé avec la méthode GlOve. Malheuresement, aussi ici les résultats sont assez mauvais, mais améliorables en changeant quelque paramètre.
