import express from 'express';
import { ChromaClient } from 'chromadb';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { performance } from 'perf_hooks';

import { 
    encodePersonFeatures, 
    trainModelOnData, 
    saveModelInstance,
    loadSavedModel, 
    predict, 
    ALLOWED_COLORS, 
    ALLOWED_LOCATIONS,
    TIERS 
} from './NeuralModels.js';

const app = express();
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Rotas de arquivos estáticos para o Frontend
app.get('/app.js', (req, res) => res.sendFile(path.join(__dirname, 'app.js')));
app.use('/tfjs-vis', express.static(path.join(__dirname, 'node_modules', '@tensorflow', 'tfjs-vis', 'dist')));

const raw_data='./data'
const usersFilePath = path.join(__dirname, `${raw_data}/users.json`);
const productsFilePath = path.join(__dirname, `${raw_data}/products.json`);


// Função auxiliar para evitar cache de ponteiros na memória do Node.js
function loadUsersFromDisk() {
    return JSON.parse(fs.readFileSync(usersFilePath, 'utf8'));
}

const products = JSON.parse(fs.readFileSync(productsFilePath, 'utf8'));

// Inicialização do cliente ChromaDB usando a nova sintaxe recomendada
const client = new ChromaClient({ host: "localhost", port: 8000 });
let collection;
let activeModelInstance = null;
let trainingLogs = { loss: [], accuracy: [] };

/**
 * Sincroniza o catálogo de produtos com o ChromaDB 
 */
async function syncChromaDB() {
    try { await client.deleteCollection({ name: "production_neural_chroma" }); } catch(e){}
    collection = await client.getOrCreateCollection({ name: "production_neural_chroma" });

    const ids = products.map(p => `prod_${p.id}`);
    const documents = products.map(p => p.name);
    const metadatas = products.map(p => ({ id: p.id, category: p.category, price: p.price, color: p.color }));
    
    const categoriesList = ['eletrônicos', 'vestuário', 'calçados', 'acessórios'];
    const embeddings = products.map(p => {
        const catArr = Array(4).fill(0);
        const idx = categoriesList.indexOf(p.category);
        if(idx !== -1) catArr[idx] = 1;
        
        // Normalização baseada em R$ 350 para evitar NaN ou Infinitos no ChromaDB
        const normalizedPrice = (p.price / 350) || 0; 
        return [...catArr, normalizedPrice];
    });

    await collection.add({ ids, embeddings, metadatas, documents });
    return products.length;
}

// Treina a Rede Neural sob demanda(manualmente) e captura o tempo exato de execução
app.post('/api/train', async (req, res) => {
    try {
        trainingLogs = { loss: [], accuracy: [] };
        const startTime = performance.now();
        
        const currentUsers = loadUsersFromDisk();
        activeModelInstance = await trainModelOnData(currentUsers, (epoch, log) => {
            trainingLogs.loss.push({ x: epoch, y: log.loss });
            trainingLogs.accuracy.push({ x: epoch, y: log.acc });
        });
        
        const endTime = performance.now();
        res.json({ success: true, executionTimeSeconds: (endTime - startTime) / 1000 });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Salva o modelo de IA no diretório local
app.post('/api/save-model', async (req, res) => {
    try {
        await saveModelInstance(activeModelInstance);
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Sincroniza e recarrega os vetores novos - ChromaDB
app.post('/api/sync-database', async (req, res) => {
    try {
        const total = await syncChromaDB();
        res.json({ success: true, totalIndexed: total });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Load do arquivo HTML - User interface
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'index.html')));

// Envia dados iniciais para preencher as caixas de seleção (selects)
app.get('/api/initial-data', (req, res) => {
    const currentUsers = loadUsersFromDisk();
    res.json({
        users: currentUsers.map(u => ({ id: u.id, name: u.name })),
        products: products.map(p => ({ id: p.id, name: p.name })),
        hasSavedModel: activeModelInstance !== null
    });
});

// Retorna o histórico de erros do treinamento para alimentar o TFJS-Vis
app.get('/api/train-history', (req, res) => res.json(trainingLogs));

// Pipeline Principal - Faz a Predição de Perfil e a Consulta no ChromaDB
app.get('/api/recommendations/:userId', async (req, res) => {
    try {
        const userId = parseInt(req.params.userId);
        const currentUsers = loadUsersFromDisk();
        const user = currentUsers.find(u => u.id === userId);
        if (!user) return res.status(404).json({ error: "Usuário inválido" });

        if (!activeModelInstance) {
            return res.json({ user: user.name, predictedTier: "PENDENTE", confidences: [], purchases: user.purchases, recommendations: [] });
        }

        const hasPurchases = user.purchases && user.purchases.length > 0;
        const samplePurchase = hasPurchases ? user.purchases[0] : { color: 'preto' };
        const userColor = ALLOWED_COLORS.includes(samplePurchase.color) ? samplePurchase.color : 'preto';
        const userLoc = ALLOWED_LOCATIONS[userId % ALLOWED_LOCATIONS.length];
        
        // ENCODER - VECTORIZAÇÃO
        // Codificação do Usuário (14 posições)
        const encodedInput = encodePersonFeatures(user.age, userColor, userLoc);
        
        // Executa Predição de Probabilidade via Softmax
        const predictions = await predict(activeModelInstance, [encodedInput]);
        
        // Cópia das predições antes do sort() para não corromper os índices categóricos
        // A predições estavam sempre empty na tela (frontend)
        const sortedPreds = [...predictions].sort((a, b) => b.prob - a.prob);
        const bestTierIndex = sortedPreds[0].index;
        const predictedTierName = TIERS[bestTierIndex];

        // Estruturação do Mapa de Métricas de Confiança para o Frontend
        const confidenceMetrics = predictions.map((p, idx) => ({
            tier: TIERS[idx].toUpperCase(),
            confidence: (p.prob * 100).toFixed(1) + '%'
        }));

        // Classificação do Cliente
        // Define o perfile do cliente baseado na decisão/recomendação da Rede Neural
        let targetPriceNorm = 0.9; // Premium Target
        if (predictedTierName === 'medium') targetPriceNorm = 0.4;
        if (predictedTierName === 'basic') targetPriceNorm = 0.1;

        // Executa a Busca por Similaridade no ChromaDB 
        const searchResults = await collection.query({
            queryEmbeddings: [[0.25, 0.25, 0.25, 0.25, targetPriceNorm]],
            nResults: 5
        });

        const purchasedIds = user.purchases.map(p => p.id);
        const recommendedProducts = [];

        // Filtra itens duplicados (já comprados)
        for (let i = 0; i < searchResults.ids[0].length; i++) {
            const metadata = searchResults.metadatas[0][i];
            if (!purchasedIds.includes(metadata.id)) {
                recommendedProducts.push({ 
                    name: searchResults.documents[0][i], 
                    ...metadata, 
                    distance: searchResults.distances[0][i] 
                });
            }
        }

        res.json({
            user: user.name,
            predictedTier: predictedTierName.toUpperCase(),
            confidences: confidenceMetrics,
            purchases: user.purchases,
            recommendations: recommendedProducts
        });
    } catch (err) {
        console.error("Erro na rota de recomendações:", err);
        res.status(500).json({ error: err.message });
    }
});

//  Simulação da compra. 
//  Atualiza a memória e re-grava o arquivo JSON
app.post('/api/simulate-purchase', (req, res) => {
    try {
        const { userId, productId } = req.body;
        const currentUsers = loadUsersFromDisk();
        const user = currentUsers.find(u => u.id === parseInt(userId));
        const product = products.find(p => p.id === parseInt(productId));

        if (user && product) {
            if (!user.purchases.some(p => p.id === product.id)) {
                user.purchases.push(product);
                // Escrita atômica segura no disco rígido local
                fs.writeFileSync(usersFilePath, JSON.stringify(currentUsers, null, 2));
            }
            res.json({ success: true });
        } else {
            res.status(400).json({ error: "Dados inválidos ou incompletos" });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Inicialização Principal do Sistema
async function start() {
    await syncChromaDB();
    activeModelInstance = await loadSavedModel();
    if (activeModelInstance) {
        console.log("⚡ Modelo recuperado e carregado do disco com sucesso.");
        // Logs  mockados para a primeira exibição na tela/página  antes de um re-treino manual
        trainingLogs.loss = [{ x: 0, y: 0.05 }, { x: 100, y: 0.005 }];
        trainingLogs.accuracy = [{ x: 0, y: 0.92 }, { x: 100, y: 0.99 }];
    }
    app.listen(3000, () => console.log("🚀 Sistema ativo em: http://localhost:3000"));
}

start().catch(console.error);
