# RNNclinics
Dans le code "RNN+LSTM+word2vec" on trouve un modèle qui entraine une LSTM (les détails modèle du réseau utilisé ici ne sont pas importants) en utilisant un word embedding issu d’une très grande base de données (plus que 3 milliards de mots) pre-entrainé par Google. Néanmoins, les résultats obtenus après l’entrainement du modèle sont très médiocres... Ceci est aussi cause du fait que la LSTM a seulement 16 unités, les epochs sont seulement 6, les séquences sont coupées à 30 mots. Tout cela peut être amélioré.


 
Dans le code de « Word2Vec model on IMDB reviews »  j’ai crée moi-même un modelé word2vec à partir des textes des reviews. Je n'ai pas encore testé le modèle ainsi obtenu sur le problème de prédiction des sentiments.



