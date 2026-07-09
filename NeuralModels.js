import * as tf from '@tensorflow/tfjs';
import fs from 'fs';

export const ALLOWED_COLORS = ['preto', 'prata', 'azul', 'vermelho', 'branco', 'bege', 'cinza', 'marrom'];
export const ALLOWED_LOCATIONS = ['SP', 'RJ', 'MG', 'PR', 'RS'];
export const TIERS = ['premium', 'medium', 'basic'];
const MODEL_SAVE_PATH = 'file://./saved_neural_model';

export function encodePersonFeatures(age, favoriteColor, location) {
    const normAge = age / 100;
    
    const colorIdx = ALLOWED_COLORS.indexOf(favoriteColor);
    const colorArr = Array(ALLOWED_COLORS.length).fill(0);
    if (colorIdx !== -1) colorArr[colorIdx] = 1;

    const locIdx = ALLOWED_LOCATIONS.indexOf(location);
    const locArr = Array(ALLOWED_LOCATIONS.length).fill(0);
    if (locIdx !== -1) locArr[locIdx] = 1;

    return [normAge, ...colorArr, ...locArr];
}

export function getProductTierIndex(price) {
    // CLASSIFICAÇÃO DO CLIENTE
    
    if (price >= 130) return 0; // premium
    if (price >= 70)  return 1; // medium
    return 2;                   // basic
}

export async function trainModelOnData(usersData, onEpochEndCallback) {
    const rawInputs = [];
    const rawOutputsIndices = [];

    usersData.forEach(u => {
        if (u.purchases && u.purchases.length > 0) {
            u.purchases.forEach(p => {
                const favColor = ALLOWED_COLORS.includes(p.color) ? p.color : 'preto';
                const simulatedLoc = ALLOWED_LOCATIONS[u.id % ALLOWED_LOCATIONS.length];
                rawInputs.push(encodePersonFeatures(u.age, favColor, simulatedLoc));
                rawOutputsIndices.push(getProductTierIndex(p.price));
            });
        }
    });

    if (rawInputs.length === 0) {
        throw new Error("Não há dados de compras suficientes no histórico para treinar o modelo.");
    }

    const numFeatures = rawInputs[0].length; // Pega o valor fixo de 14 colunas
    const numClasses = 3;

    const { inputXs, outputYs } = tf.tidy(() => {
        const inputsTensor = tf.tensor2d(rawInputs);
        const indicesTensor = tf.tensor1d(rawOutputsIndices, 'int32');
        const outputsTensor = tf.oneHot(indicesTensor, numClasses);
        return { inputXs: inputsTensor, outputYs: outputsTensor };
    });

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [numFeatures], units: 80, activation: 'relu' }));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(inputXs, outputYs, {
        verbose: 0,
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                if (onEpochEndCallback) onEpochEndCallback(epoch, log);
            }
        }
    });

    inputXs.dispose();
    outputYs.dispose();

    return model;
}

/**
 * Salva explicitamente a instância atual do modelo no disco
 */
export async function saveModelInstance(model) {
    if (!model) throw new Error("Nenhum modelo ativo na memória para ser salvo.");
    if (!fs.existsSync('./saved_neural_model')) {
        fs.mkdirSync('./saved_neural_model');
    }
    await model.save(MODEL_SAVE_PATH);
    console.log("💾 Instância do modelo persistida no disco.");
}

export async function loadSavedModel() {
    if (fs.existsSync('./saved_neural_model/model.json')) {
        return await tf.loadLayersModel(`${MODEL_SAVE_PATH}/model.json`);
    }
    return null;
}

export async function predict(model, pessoa) {
    const tfInput = tf.tensor2d(pessoa);
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    
    tfInput.dispose();
    pred.dispose();
    
    return predArray.map((prob, index) => ({ prob, index }));
}
