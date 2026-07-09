const userSelect = document.getElementById('userSelect');
const productSelect = document.getElementById('productSelect');
const buyBtn = document.getElementById('buyBtn');
const trainBtn = document.getElementById('trainBtn');
const saveBtn = document.getElementById('saveBtn');
const dbBtn = document.getElementById('dbBtn');
const trainStatus = document.getElementById('trainStatus');
const accuracyLabel = document.getElementById('accuracyLabel');
const timeLabel = document.getElementById('timeLabel');

async function initPage() {
    const res = await fetch('/api/initial-data');
    const data = await res.json();
    
    data.users.forEach(u => {
        const opt = document.createElement('option'); opt.value = u.id; opt.innerText = u.name;
        userSelect.appendChild(opt);
    });

    data.products.forEach(p => {
        const opt = document.createElement('option'); opt.value = p.id; opt.innerText = p.name;
        productSelect.appendChild(opt);
    });

    userSelect.addEventListener('change', (e) => onUserChange(e.target.value));
    buyBtn.addEventListener('click', submitSimulatedPurchase);
    trainBtn.addEventListener('click', executeModelTraining);
    saveBtn.addEventListener('click', executeModelSaving);
    dbBtn.addEventListener('click', executeDBSync);

    trainStatus.innerText = data.hasSavedModel ? "Modelo carregado do disco." : "Aguardando primeiro treino.";
    if(data.hasSavedModel) saveBtn.disabled = false;
    await refreshCharts();
}

function onUserChange(userId) {
    const hasUser = userId !== "";
    productSelect.disabled = !hasUser;
    buyBtn.disabled = !hasUser;
    if (hasUser) loadDashboard(userId);
}

async function refreshCharts() {
    const trainRes = await fetch('/api/train-history');
    const historyData = await trainRes.json();
    if (historyData.accuracy && historyData.accuracy.length > 0) {
        const finalAcc = historyData.accuracy[historyData.accuracy.length - 1].y * 100;
        accuracyLabel.innerText = `Acurácia Final: ${finalAcc.toFixed(2)}%`;
        tfvis.render.linechart(document.getElementById('perf-loss'), { values: historyData.loss }, { xLabel: 'Epoch', yLabel: 'Loss', height: 220 });
        tfvis.render.linechart(document.getElementById('perf-acc'), { values: historyData.accuracy }, { xLabel: 'Epoch', yLabel: 'Accuracy', height: 220 });
    }
}

async function executeModelTraining() {
    trainStatus.innerText = "⏳ Treinando rede..."; trainStatus.style.color = "#ecc94b";
    trainBtn.disabled = true; saveBtn.disabled = true;

    const res = await fetch('/api/train', { method: 'POST' });
    const data = await res.json();

    trainStatus.innerText = "✅ Treino concluído! (Ainda não salvo)"; trainStatus.style.color = "#007bff";
    timeLabel.innerText = `Tempo: ${data.executionTimeSeconds.toFixed(3)}s`;
    
    trainBtn.disabled = false; saveBtn.disabled = false;
    await refreshCharts();
    if (userSelect.value) loadDashboard(userSelect.value);
}

async function executeModelSaving() {
    trainStatus.innerText = "💾 Salvando arquivos no disco...";
    const res = await fetch('/api/save-model', { method: 'POST' });
    const data = await res.json();
    if(data.success) {
        trainStatus.innerText = "✅ Arquivos gravados na pasta saved_neural_model/!";
        trainStatus.style.color = "#48bb78";
    }
}

async function executeDBSync() {
    dbBtn.disabled = true;
    trainStatus.innerText = "⏳ Atualizando índices no ChromaDB...";
    const res = await fetch('/api/sync-database', { method: 'POST' });
    const data = await res.json();
    if(data.success) {
        trainStatus.innerText = `✅ ChromaDB Atualizado! (${data.totalIndexed} produtos indexados)`;
        trainStatus.style.color = "#48bb78";
        if (userSelect.value) loadDashboard(userSelect.value);
    }
}

async function loadDashboard(userId) {
    const res = await fetch(`/api/recommendations/${userId}`);
    const data = await res.json();

    // Renderização o histórico de compras do cliente
    let pList = '';
    data.purchases.forEach(p => {
        pList += `<div class="card purchase-card"><strong>${p.name}</strong><br><span class="badge">${p.category}</span><span class="badge">${p.color}</span><div style="margin-top:6px; color:#28a745; font-weight:bold;">R$ ${p.price}</div></div>`;
    });
    document.getElementById('purchasesList').innerHTML = pList || 'Sem histórico.';

    // Bloco da Rede Neural mostrando a distribuição de confiança do Softmax - reinderização
    // obs: se quiser , trocar softmax por sigmoid
    let confidenceHtml = '';
    data.confidences.forEach(c => {
        confidenceHtml += `<span class="badge" style="background: #edf2f7; color: #4a5568;">${c.tier}: ${c.confidence}</span> `;
    });

    let rList = `
        <div class="card neural-card" style="border-left-color: #9f7aea; background: #faf5ff;">
            <strong>Distribuição de Confiança da IA:</strong>
            <div style="margin-top: 8px; margin-bottom: 5px;">${confidenceHtml}</div>
            <p style="margin: 8px 0 0 0; font-size: 14px; font-weight: bold; color: #6b46c1;">
                🎯 Recomendação Principal: Faixa de Preço ${data.predictedTier}
            </p>
        </div>
        <h4 style="margin: 20px 0 10px 0; color: #4a5568;">Sugestões do Catálogo no ChromaDB:</h4>
    `;

    //  produtos sugeridos com base no histórico do Chromadb 
    data.recommendations.forEach(r => {
        rList += `
        <div class="card recom-card" style="border-left-color: #007bff;">
            <strong>${r.name}</strong><br>
            <span class="badge">${r.category}</span>
            <span class="badge">${r.color}</span>
            <span class="badge" style="background: #ebf8ff; color: #2b6cb0;">R$ ${r.price}</span>
            <div style="margin-top:8px; font-size: 12px; color:#718096; font-weight:bold;">
                Afinidade Vetorial (Distância Cosseno): ${r.distance.toFixed(4)}
            </div>
        </div>`;
    });
    
    document.getElementById('recommendationsList').innerHTML = rList || '<div class="card">Nenhum produto correspondente encontrado no banco vetorial.</div>';
}


async function submitSimulatedPurchase() {
    const userId = userSelect.value;
    const productId = productSelect.value;
    if (!productId) return alert('Selecione um produto!');

    await fetch('/api/simulate-purchase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, productId })
    });

    dbBtn.disabled = false;
    loadDashboard(userId);
}

initPage();
