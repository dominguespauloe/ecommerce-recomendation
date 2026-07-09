# 🚀 Sistema de Recomendação Inteligente (TensorFlow.js + ChromaDB)

Este projeto implementa um pipeline completo de **Sistema de Recomendação E-commerce** baseado em Inteligência Artificial. Ele combina o poder preditivo de uma **Rede Neural Sequencial (TensorFlow.js)** para classificar o perfil econômico/comportamental do cliente com um **Banco de Dados Vetorial (ChromaDB)** para buscar produtos por similaridade cossena em tempo real.

A aplicação conta com uma interface web interativa rica onde é possível simular compras, atualizar dados vetoriais, treinar a rede neural sob demanda e acompanhar gráficos de acurácia/perda na tela através do **TFJS-Vis**.

---

## 🏗️ Arquitetura do Projeto e Estrutura de Pastas

O projeto adota uma arquitetura modular desacoplada seguindo as melhores práticas de engenharia de software:

```text
├── NeuralModels.js      # Core de IA: Engenharia de Atributos, Treino e Predição (TFJS)
├── server.js            # Servidor Backend (Express) e gerenciamento de APIs Restful
├── index.html           # Interface Gráfica da aplicação (Frontend)
├── app.js               # Lógica de interação da tela e requisições HTTP (Frontend)
├── users.json           # Banco de dados estruturado em JSON para Usuários/Clientes
├── products.json        # Catálogo de produtos estruturado em JSON
├── saved_neural_model/  # Pasta gerada automaticamente ao salvar os pesos da IA
└── package.json         # Dependências do ecossistema Node.js
```

---

## 💻 Detalhamento do Código e Funcionamento

### 1. Engenharia de Recursos & Vetorização (`NeuralModels.js`)
* **`encodePersonFeatures` (Entrada de 14 Posições):** Converte a idade normalizada do cliente ($idade / 100$) combinada com uma codificação *One-Hot* de 8 posições para cores prediletas e 5 posições para localizações geográficas. Totalizando uma matriz de entrada rígida de 14 dimensões.
* **`getProductTierIndex` (Saída de 3 Categorias):** Atua como uma árvore de decisão de negócio baseada em faixas de preço reais (Preço $\ge$ 130 = Premium [Índice 0], Preço $\ge$ 70 = Medium [Índice 1], senão Basic [Índice 2]).
* **`trainModelOnData` (Rede Neural):** Instancia um modelo sequencial denso com **80 neurônios na camada oculta** e ativação `ReLU` (filtra e propaga apenas informações relevantes). A camada de saída utiliza ativação `Softmax`, normalizando os resultados em distribuições de probabilidade numéricas exatas somando 100%. O otimizador utilizado é o `Adam`, calculando os erros através da função de perda `categoricalCrossentropy`.
* **Mecanismo de Persistência:** Utiliza o protocolo nativo `file://` do TensorFlow para gravar a topologia binária diretamente na pasta `saved_neural_model/` de forma persistente.

### 2. Fluxo de Recomendações e Rotas (`server.js`)
* O servidor Express centraliza as requisições eliminando caches mutáveis indesejados ao recarregar arquivos JSON atonicamente via `fs.readFileSync` no momento de cada chamada.
* **Média de Preferências Combinada:** O backend monta o vetor do perfil histórico, realiza a predição na rede neural e clona as probabilidades geradas com operadores de espalhamento (`[...predictions].sort()`), evitando corromper a ordem dos índices originais do Softmax.
* **Alinhamento Vetorial no ChromaDB:** Com o Tier classificado (Premium, Medium ou Basic), define-se uma projeção de preço padrão e o banco executa uma busca de vizinhos mais próximos (KNN via distância cossena), retornando produtos similares enquanto filtra automaticamente itens já comprados.

---

## 🛠️ Como Executar no Windows SEM Docker (Modo Nativo)

Se você não possui ou prefere não utilizar o Docker no Windows, pode rodar o ChromaDB nativamente através do Python.

### Pré-requisitos
1. **Node.js LTS** (versão 18 ou superior) instalado.
2. **Python 3.10 ou 3.11** instalado (Certifique-se de marcar a opção *"Add Python to PATH"* durante a instalação).
3. **C++ Build Tools** instalado (exigido para compilar extensões nativas do TensorFlow/Chroma no Windows).

### Passo a Passo

**1. Instalar e rodar o ChromaDB via Python:**
Abra o PowerShell ou Prompt de Comando e instale o servidor do ChromaDB:
```bash
pip install chromadb
```
Inicie o servidor nativo especificando a porta padrão:
```bash
chroma run --host localhost --port 8000
```
*Mantenha essa janela do terminal aberta.*

**2. Configurar o projeto Node.js:**
Abra um novo terminal na pasta raiz do projeto e instale todas as dependências declaradas:
```bash
npm install
```

**3. Iniciar o Servidor Backend:**
Inicie a aplicação utilizando o interpretador do Node:
```bash
node server.js
```

**4. Acessar a Interface:**
Abra o navegador de sua preferência e acesse o endereço: **`http://localhost:3000`**

---

## 🐳 Como Executar com Docker (Recomendado)

O uso do Docker isola o banco vetorial, eliminando a necessidade de instalar compiladores C++ ou Python diretamente no seu Windows.

### Pré-requisitos
1. **Node.js LTS** instalado localmente na máquina.
2. **Docker Desktop** instalado e rodando no Windows.

### Passo a Passo

**1. Baixar e Rodar o Contêiner do ChromaDB:**
Abra o terminal do Windows e execute o comando abaixo. Ele fará o download da imagem oficial e montará um volume local garantindo que seus vetores salvos não desapareçam ao desligar o contêiner:
```bash
docker run -d --name chromadb -p 8000:8000 -v ./chroma-data:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=FALSE chromadb/chroma:latest
```
*(Caso o contêiner já tenha sido criado anteriormente, basta executar apenas `docker start chromadb`)*.

**2. Verificar se o Banco está Ativo:**
Confirme que o contêiner está rodando em segundo plano:
```bash
docker ps
```

**3. Configurar e Iniciar o Node.js:**
Na pasta do seu projeto, instale as dependências locais e suba o servidor de aplicação:
```bash
npm install
node server.js
```

**4. Acessar a Interface:**
Acesse o painel administrativo através do endereço: **`http://localhost:3000`**

---

## 🕹️ Guia de Uso da Interface Web

1. **Seleção de Clientes:** Escolha um usuário na primeira listagem (ex: *Ana Lima*). O sistema exibirá o histórico atual em tempo real na coluna esquerda.
2. **Treinamento Gráfico:** Clique no botão azul **"⚙️ Treinar Rede Neural"**. O painel superior mostrará o tempo exato decorrido (em milissegundos/segundos) e os gráficos de linha do **TFJS-Vis** serão desenhados dinamicamente na lateral direita exibindo a curva de acurácia chegando perto de 100%.
3. **Persistência de IA:** Clique no botão laranja **"💾 Salvar Modelo Local (Disco)"** para gravar a rede atual. Ao reiniciar o servidor futuramente, ele lerá este arquivo instantaneamente sem necessitar de novos treinos.
4. **Simulação Dinâmica:** Escolha um produto qualquer no segundo campo de seleção e clique em **"Comprar"**. O item entra no histórico do cliente.
5. **Sincronização Vetorial:** Após simular compras, o botão roxo **"💾 Salvar Novos Vetores no ChromaDB"** ficará disponível. Ao clicar nele, a árvore estrutural do banco vetorial no Docker é reindexada com as novas coordenadas de afinidade automaticamente.


## ☁️ Como Executar no Google Cloud Platform (GCP) com Kubernetes (GKE)

Para escalar este sistema de recomendação em produção, utilizaremos o **

https://cloud.google.com/kubernetes-engine

** para orquestrar dois Pods principais: o servidor da sua aplicação Node.js e o banco vetorial ChromaDB, garantindo persistência de dados via *Persistent Volumes*.

### 🏗️ Arquitetura Kubernetes para o Projeto

```text
               ┌──────────────── GKE Cluster ────────────────┐
               │                                             │
               │   ┌──────────────┐          ┌───────────┐   │
Ingress / ────┼──►│  Node.js App │─────────►│ ChromaDB  │   │
LoadBalancer   │   │  (Port 3000) │          │ (Port 8000│   │
               │   └──────────────┘          └─────┬─────┘   │
               │                                   │         │
               │                                   ▼         │
               │                           ┌───────────────┐ │
               │                           │  Persistent   │ │
               │                           │  Volume (GCS) │ │
               │                           └───────────────┘ │
               └─────────────────────────────────────────────┘
```

### 📋 Pré-requisitos
1. **Google Cloud SDK (gcloud CLI)** instalado e autenticado na sua máquina Windows.
2. Componente `kubectl` instalado (`gcloud components install kubectl`).
3. Um projeto ativo no GCP com o faturamento habilitado e a API do Kubernetes Engine ativada.

---

### Passo 1: Containerizar a Aplicação Node.js

Crie um arquivo chamado `Dockerfile` na raiz do seu projeto para empacotar o backend:

```dockerfile
FROM node:18-slim

# Instala ferramentas necessárias para compilação de pacotes nativos do TFJS no Linux
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install --only=production

COPY . .

EXPOSE 3000
CMD [ "node", "server.js" ]
```

Suba a imagem para o **Artifact Registry** ou **Container Registry** do seu projeto no Google Cloud:
```bash
# Autentique o Docker no GCP
gcloud auth configure-docker

# Construa a imagem taggeando com o ID do seu projeto GCP
docker build -t gcr.io/[ID-DO-SEU-PROJETO]/recommendation-app:latest .

# Envie a imagem para a nuvem
docker push gcr.io/[ID-DO-SEU-PROJETO]/recommendation-app:latest
```

---

### Passo 2: Criar o Cluster Kubernetes no GKE

Execute o comando abaixo para criar um cluster gerenciado padrão com 2 nós para suportar o processamento do TensorFlow.js e do ChromaDB:

```bash
gcloud container clusters create recommendation-cluster \
    --num-nodes=2 \
    --zone=us-central1-a \
    --machine-type=e2-standard-2
```

Configure o seu terminal local para apontar para o novo cluster criado:
```bash
gcloud container clusters get-credentials recommendation-cluster --zone us-central1-a
```

---

### Passo 3: Criar os Manifestos de Implantação do Kubernetes

Crie uma pasta chamada `k8s/` na raiz do projeto e adicione os três arquivos de configuração descritos abaixo:

#### 1. Persistência de Dados (`k8s/chroma-pvc.yaml`)
Garante que os vetores do ChromaDB permaneçam salvos em discos persistentes do Google Cloud, mesmo que o contêiner reinicie ou sofra manutenção.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard-rwo # Provisiona um disco permanente do GCP de forma nativa
```

#### 2. Infraestrutura do ChromaDB (`k8s/chromadb-deployment.yaml`)
Configura o Pod do ChromaDB e o expõe internamente na rede do cluster através de um Service do Kubernetes na porta 8000.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chromadb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chromadb
  template:
    metadata:
      labels:
        app: chromadb
    spec:
      containers:
      - name: chromadb
        image: chromadb/chroma:latest
        ports:
        - containerPort: 8000
        env:
        - name: IS_PERSISTENT
          value: "TRUE"
        - name: ANONYMIZED_TELEMETRY
          value: "FALSE"
        volumeMounts:
        - name: chroma-storage
          mountPath: /chroma/chroma
      volumes:
      - name: chroma-storage
        persistentVolumeClaim:
          claimName: chroma-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: chromadb-service
spec:
  selector:
    app: chromadb
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

#### 3. Infraestrutura da Aplicação Node.js (`k8s/app-deployment.yaml`)
Subirá sua aplicação Node.js conectando-a ao serviço interno do ChromaDB através da variável de ambiente, expondo o painel visual para a internet pública através de um balanceador de carga (`LoadBalancer`).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: recommendation-app
  template:
    metadata:
      labels:
        app: recommendation-app
    spec:
      containers:
      - name: app
        image: gcr.io/[ID-DO-SEU-PROJETO]/recommendation-app:latest # Substitua pelo ID do seu projeto
        ports:
        - containerPort: 3000
        env:
        # Aponta o Host do ChromaDB para o DNS interno do Service criado anteriormente
        - name: CHROMA_HOST
          value: "chromadb-service" 
        - name: CHROMA_PORT
          value: "8000"
---
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  type: LoadBalancer # Cria um endereço de IP Público no Google Cloud para você acessar
  selector:
    app: recommendation-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
```

---

### Passo 4: Aplicar os arquivos no Cluster GKE

Envie todas as configurações da pasta para a esteira de execução do cluster Kubernetes:

```bash
kubectl apply -f k8s/
```

### Passo 5: Acompanhar o Status e Acessar a URL do GCP

Para verificar se os contêineres subiram com sucesso na nuvem, execute:
```bash
kubectl get pods
```

Para encontrar o endereço IP externo gerado pelo Google Cloud para o seu painel de Inteligência Artificial, execute o comando de inspeção de serviços:
```bash
kubectl get services app-service
```

Aguarde o status da coluna `EXTERNAL-IP` mudar de `<pending>` para um IP válido (ex: `35.224.12.34`). Copie esse endereço e cole-o no seu navegador para utilizar o seu sistema de recomendação operando em escala de produção na nuvem!


// No seu arquivo server.js substitua a inicialização por:
const chromaHost = process.env.CHROMA_HOST || "localhost";
const chromaPort = parseInt(process.env.CHROMA_PORT) || 8000;

const client = new ChromaClient({ host: chromaHost, port: chromaPort });

## 📈 Configuração de Escalonamento Automático (Autoscaling com HPA)

Para garantir que o sistema não caia ou fique lento quando dezenas de administradores clicarem no botão "Treinar Rede Neural" simultaneamente, utilizaremos o recurso nativo **Horizontal Pod Autoscaler (HPA)** do Kubernetes.

O HPA monitora os Pods em tempo real e, se o consumo médio de processamento ultrapassar um limite definido, ele instrui o cluster do Google Cloud a criar novas réplicas da aplicação imediatamente.

### 📋 Pré-requisitos
Para que o HPA funcione no GKE, o cluster precisa ter o **Metrics Server** ativo (ativado por padrão no GKE) e os Pods **devem** ter limites mínimos e máximos de recursos declarados explicitamente em seu arquivo de Deployment.

---

### Passo 1: Atualizar o arquivo `k8s/app-deployment.yaml`

Abra o seu arquivo `k8s/app-deployment.yaml` e atualize a seção de configuração do contêiner (`containers:`) para incluir as definições de `resources`. Isso é obrigatório para o Kubernetes calcular as porcentagens de estresse do sistema.

Substitua a estrutura do Deployment por esta:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-app
spec:
  replicas: 1 # Começa com apenas 1 Pod para economizar recursos
  selector:
    matchLabels:
      app: recommendation-app
  template:
    metadata:
      labels:
        app: recommendation-app
    spec:
      containers:
      - name: app
        image: gcr.io/[ID-DO-SEU-PROJETO]/recommendation-app:latest
        ports:
        - containerPort: 3000
        env:
        - name: CHROMA_HOST
          value: "chromadb-service" 
        - name: CHROMA_PORT
          value: "8000"
        
        # DEFINIÇÃO DE LIMITES (Obrigatório para o HPA operar)
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"   # Garante 0.2 vCPU mínimo para iniciar o Node
          limits:
            memory: "1Gi"  # Margem segura para manipulação de tensores na memória
            cpu: "800m"   # Evita que um treino infinito trave a máquina do Google inteira
---
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  type: LoadBalancer
  selector:
    app: recommendation-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
```

---

### Passo 2: Criar a Política de Autoscaling (`k8s/app-hpa.yaml`)

Crie um novo arquivo dentro da sua pasta de manifestos chamado `k8s/app-hpa.yaml`. Este arquivo instrui o Kubernetes a escalar a sua aplicação de **1 até 5 réplicas** caso o uso de CPU média passe de **70%**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: app-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recommendation-app # Alvo que será clonado dinamicamente
  minReplicas: 1
  maxReplicas: 5 # Limite máximo de clones para evitar custos descontrolados no GCP
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70 # Dispara se a CPU passar de 70% do alocado (140m)
```

---

### Passo 3: Aplicar as Novas Regras no GKE

Envie as novas configurações de recursos e a política do HPA para a nuvem rodando:

```bash
kubectl apply -f k8s/
```

---

### 🕹️ Como Monitorar o Escalonamento na Tela

Para ver a Inteligência Artificial do Kubernetes agindo em tempo real e criando clones do seu servidor conforme o estresse matemático do TensorFlow.js acontece, abra uma nova janela de terminal e execute o comando de monitoramento contínuo:

```bash
kubectl get hpa app-autoscaler --watch
```

A saída exibirá um relatório dinâmico semelhante a este:

```text
NAME             REFERENCE                       TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
app-autoscaler   Deployment/recommendation-app   5%/70%    1         5         1          2m
app-autoscaler   Deployment/recommendation-app   98%/70%   1         5         1          4m  <-- Alguém iniciou um treino!
app-autoscaler   Deployment/recommendation-app   98%/70%   1         5         3          5m  <-- Kubernetes criando +2 Pods!
app-autoscaler   Deployment/recommendation-app   32%/70%   1         5         3          8m  <-- Carga distribuída com sucesso.
```

Quando o processamento do TensorFlow.js terminar e os Pods ficarem ociosos por alguns minutos, o Kubernetes fará o processo inverso (*Scale Down*), destruindo as réplicas extras de forma segura e voltando para apenas **1 Pod**, garantindo economia financeira automatizada na sua fatura do Google Cloud.
