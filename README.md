# @return - É pra comprar ou pra vender?
## Intro
Este é o **@return**, um projeto pessoal, programado em **python/django** que surgiu com uma necessidade minha (que gosto muito do mercado de ações) de tentar prever se no dia corrente devo **comprar** ou **vender** determinado papel (ação). 
## Como funciona
Para consultar a previsão dada pela aplicação basta digitar no campo **ação** o código do papel que deseja consultar. Ex. para Petrobrás podemos usar PETR4. Há possibilidade de escolha do período da base histórica e do algoritmo (caso não deseje utilizar todos).
## Algoritmos utilizados
Utilizei 5 algoritmos com abordagens diferentes para calcular as previsões. São eles:
1. Naive Bayes
2. Decision Tree
3. Random Forest
4. KNN
5. SVC

Na tela dos resultados serão apresentados separadamente o cálculo de cada algoritmo que poderá ser detalhada com a probabilidade de exatidão daquela previsão considerando a base histórica escolhida.


