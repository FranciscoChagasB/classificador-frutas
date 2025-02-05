# 🍏🍌🍇 Classificador de Frutas  

Este projeto utiliza aprendizado de máquina para classificar imagens de frutas.  
Ele foi treinado com o dataset [Fruits 100 do Kaggle](https://www.kaggle.com/datasets/marquis03/fruits-100).  
A acurácia do modelo foi de 92.5722

## ⚠️ Aviso  
O problema foi simplificado para considerar apenas as seguintes frutas: **maçã, banana, uva, manga e abacaxi**.  
Caso queira trabalhar com o dataset reduzido, você pode baixá-lo do meu Google Drive:  
- **Test:** [Link](https://drive.google.com/drive/folders/19M3NpiRHtuOs6DPVogfJQafYM6o14Pt3?usp=sharing)  
- **Train:** [Link](https://drive.google.com/drive/folders/1ulrduvbzPwCgun22BW0-B85AiDIA6g5t?usp=sharing)  
Se preferir utilizar todas as classes do dataset original, será necessário modificar o arquivo `classifier_images.py` no trecho:  
```python
# Classes de frutas
target_classes = ['maçã', 'banana', 'uva', 'manga', 'abacaxi']
```
Substitua essa lista pelas classes desejadas.  

## 📂 Estrutura do Projeto  

```
📁 projeto/
│── 📄 train_model.py        # Script para treinar o modelo
│── 📄 classifier_images.py   # Script para classificar novas imagens
│── 📁 train/                   # Pasta com imagens de treino (baixar do Kaggle ou Drive)
│── 📁 test/                    # Pasta com imagens de teste (baixar do Kaggle ou Drive)
│── 📁 imagens_para_classificar/ # Adicione imagens para classificação
│── 📁 output/                   # Resultado das imagens classificadas
```

## 🚀 Como Usar  

### 1️⃣ Baixar o dataset  
Baixe o conjunto de dados do Kaggle no link:  
🔗 [Fruits 100 Dataset](https://www.kaggle.com/datasets/marquis03/fruits-100)  
Ou utilize o dataset reduzido disponível no Google Drive (links acima).  

### 2️⃣ Organizar os arquivos  
Coloque as pastas `train/` e `test/` no diretório do projeto.  

### 3️⃣ Instalar dependências  
Antes de rodar os scripts, instale as bibliotecas necessárias:  
```bash
pip install tensorflow keras numpy opencv-python
```

### 4️⃣ Treinar o modelo  
Execute o script para treinar o modelo com as imagens:  
```bash
python train_model.py
```

### 5️⃣ Classificar imagens  
Coloque as imagens na pasta `imagens_para_classificar/` e execute:  
```bash
python classifier_images.py
```
As imagens classificadas serão salvas na pasta `output/`.  

## 📊 Resultados  

Após o treinamento, o modelo poderá distinguir entre as frutas selecionadas com alta precisão, baseado no conjunto de dados fornecido.  

## 📜 Licença  

Este projeto é de livre uso para fins educacionais e experimentação. 🚀
