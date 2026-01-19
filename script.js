const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

const feedbackElement = document.getElementById('feedback');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const liveBadge = document.getElementById('liveBadge');
const liveMeta = document.getElementById('liveMeta');

const autoGoodBtn = document.getElementById('autoGoodBtn');
const autoBadBtn = document.getElementById('autoBadBtn');
const cancelAutoBtn = document.getElementById('cancelAutoBtn');
const trainBtn = document.getElementById('trainBtn');
const accuracyBtn = document.getElementById('accuracyBtn');
const exportBtn = document.getElementById('exportBtn');
const importBtn = document.getElementById('importBtn');
const clearBtn = document.getElementById('clearBtn');

const countGoodEl = document.getElementById('countGood');
const countBadEl = document.getElementById('countBad');
const countTotalEl = document.getElementById('countTotal');
const minPerLabelEl = document.getElementById('minPerLabel');

const LABEL_GOOD = 'goede_houding';
const LABEL_BAD = 'slechte_houding';

const config = {
  minSamplesPerLabel: 20,
  countdownSeconds: 3,
  trainSplit: 0.8,
  autoBatchCount: 20
};

let knnClassifier;
let savedPoses = [];
let isClassifierReady = false;
let currentLandmarks = null;

let countdownActive = false;
let countdownTimerId = null;
let countdownLabel = null;
let countdownRemaining = config.countdownSeconds;

let trainSet = [];
let testSet = [];
let hasTrainedModel = false;

let cmEls = null;

/* Batch auto-sampling state */
let batchActive = false;
let batchLabel = null;
let batchTotal = 0;
let batchDone = 0;

document.addEventListener('DOMContentLoaded', init);

function init() {
  if (!videoElement || !canvasElement || !feedbackElement) {
    console.error('Essentiële DOM elementen ontbreken. Check je index.html ids.');
    return;
  }

  if (minPerLabelEl) minPerLabelEl.textContent = String(config.minSamplesPerLabel);

  autoGoodBtn.addEventListener('click', () => startBatchAutoSampling(LABEL_GOOD));
  autoBadBtn.addEventListener('click', () => startBatchAutoSampling(LABEL_BAD));
  cancelAutoBtn.addEventListener('click', cancelAllSampling);
  trainBtn.addEventListener('click', trainModel);
  accuracyBtn.addEventListener('click', calculateTestMetrics);
  exportBtn.addEventListener('click', exportToJSON);
  importBtn.addEventListener('click', importFromJSON);
  clearBtn.addEventListener('click', clearData);

  ensureConfusionMatrixUI();

  setButtonsDuringSampling(false);
  setStatus('Opstarten...', 'warn');
  setBadgeState('neutral');
  liveBadge.textContent = 'Live';
  liveMeta.textContent = 'Geen voorspelling';

  setFeedback('Start de houding detectie...', 'neutral');
  updateCountsUI();
  resetMatrixUI();

  startPoseDetection();
  initializeML5();
}

function initializeML5() {
  if (typeof ml5 === 'undefined') {
    setStatus('ML5 niet geladen', 'bad');
    setFeedback('ML5 is niet geladen. Check de script tag in index.html.', 'bad');
    return;
  }

  try {
    knnClassifier = ml5.KNNClassifier();
    isClassifierReady = true;
    setStatus('AI klaar', 'good');
    setFeedback('AI klaar. Neem samples en klik daarna op Train Model.', 'good');
  } catch (error) {
    console.error('Fout bij initialiseren ML5:', error);
    setStatus('AI fout', 'bad');
    setFeedback('Kon AI niet initialiseren.', 'bad');
  }
}

async function startPoseDetection() {
  const pose = new Pose({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });

  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults(onResults);

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await pose.send({ image: videoElement });
    },
    width: 640,
    height: 480
  });

  camera.start();
}

function onResults(results) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (!results.poseLandmarks) {
    currentLandmarks = null;
    liveMeta.textContent = 'Geen persoon in beeld';
    setBadgeState('neutral');
    liveBadge.textContent = 'Live';
    return;
  }

  currentLandmarks = results.poseLandmarks;

  drawConnectors(canvasCtx, currentLandmarks, POSE_CONNECTIONS);
  drawLandmarks(canvasCtx, currentLandmarks);

  predictPose(currentLandmarks);
}

async function predictPose(landmarks) {
  if (!isClassifierReady) return;

  let numLabels = 0;
  try {
    numLabels = await knnClassifier.getNumLabels();
  } catch {
    numLabels = 0;
  }

  if (numLabels === 0 || !hasTrainedModel) {
    liveMeta.textContent = 'Nog niet getraind';
    setBadgeState('neutral');
    liveBadge.textContent = 'Live';
    return;
  }

  const features = normalizePose(landmarks).flat();

  try {
    const result = await knnClassifier.classify(features);

    liveMeta.textContent = `Voorspelling: ${result.label}`;

    if (result.label === LABEL_GOOD) {
      liveBadge.textContent = 'Goed';
      setBadgeState('good');
    } else if (result.label === LABEL_BAD) {
      liveBadge.textContent = 'Slecht';
      setBadgeState('bad');
    } else {
      liveBadge.textContent = 'Live';
      setBadgeState('neutral');
    }
  } catch (error) {
    console.error('Classificatie fout:', error);
  }
}

function normalizePose(landmarks) {
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

  const centerX = (leftHip.x + rightHip.x) / 2;
  const centerY = (leftHip.y + rightHip.y) / 2;

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];

  const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;

  const torsoSize = Math.hypot(shoulderX - centerX, shoulderY - centerY) || 1;

  return landmarks.map(lm => [
    (lm.x - centerX) / torsoSize,
    (lm.y - centerY) / torsoSize,
    lm.z / torsoSize
  ]);
}

/* ===========================
   BATCH AUTO-SAMPLING (20x)
   =========================== */

function startBatchAutoSampling(label) {
  if (batchActive || countdownActive) return;

  if (!currentLandmarks) {
    setFeedback('Geen persoon in beeld. Ga in beeld zitten en probeer opnieuw.', 'bad');
    return;
  }

  batchActive = true;
  batchLabel = label;
  batchTotal = config.autoBatchCount;
  batchDone = 0;

  setButtonsDuringSampling(true);
  setFeedback(batchStatusText('Start'), 'neutral');

  startSingleCountdownForBatch();
}

function startSingleCountdownForBatch() {
  if (!batchActive) return;

  if (!currentLandmarks) {
    finishBatchWithError('Geen pose gevonden. Batch gestopt.');
    return;
  }

  countdownActive = true;
  countdownLabel = batchLabel;
  countdownRemaining = config.countdownSeconds;

  setFeedback(batchStatusText(`Aftellen: ${countdownRemaining}`), 'neutral');

  countdownTimerId = setInterval(() => {
    countdownRemaining -= 1;

    if (countdownRemaining <= 0) {
      clearInterval(countdownTimerId);
      countdownTimerId = null;
      countdownActive = false;

      if (!currentLandmarks) {
        finishBatchWithError('Geen pose gevonden op het moment van opslaan. Batch gestopt.');
        return;
      }

      saveSample(countdownLabel);
      batchDone += 1;

      if (batchDone >= batchTotal) {
        finishBatchSuccess();
        return;
      }

      startSingleCountdownForBatch();
    } else {
      setFeedback(batchStatusText(`Aftellen: ${countdownRemaining}`), 'neutral');
    }
  }, 1000);
}

function batchStatusText(prefix) {
  const friendly = batchLabel === LABEL_GOOD ? 'goede houding' : 'slechte houding';
  return `${prefix} | Batch: ${friendly} (${batchDone + 1}/${batchTotal})`;
}

function finishBatchSuccess() {
  batchActive = false;
  batchLabel = null;
  batchTotal = 0;
  batchDone = 0;

  setButtonsDuringSampling(false);
  setFeedback('Batch klaar. Je kunt nu de andere houding opnemen of Train Model klikken.', 'good');
}

function finishBatchWithError(message) {
  batchActive = false;
  batchLabel = null;
  batchTotal = 0;
  batchDone = 0;

  if (countdownTimerId) clearInterval(countdownTimerId);
  countdownTimerId = null;
  countdownActive = false;
  countdownLabel = null;

  setButtonsDuringSampling(false);
  setFeedback(message, 'bad');
}

function cancelAllSampling() {
  if (countdownTimerId) clearInterval(countdownTimerId);
  countdownTimerId = null;

  countdownActive = false;
  countdownLabel = null;

  batchActive = false;
  batchLabel = null;
  batchTotal = 0;
  batchDone = 0;

  setButtonsDuringSampling(false);
  setFeedback('Auto-sampling gestopt.', 'neutral');
}

function setButtonsDuringSampling(isSampling) {
  cancelAutoBtn.disabled = !isSampling;
  autoGoodBtn.disabled = isSampling;
  autoBadBtn.disabled = isSampling;

  trainBtn.disabled = isSampling;
  accuracyBtn.disabled = isSampling;
  exportBtn.disabled = isSampling;
  importBtn.disabled = isSampling;
  clearBtn.disabled = isSampling;
}

/* ===========================
   TRAIN / TEST + METRICS
   =========================== */

function saveSample(label) {
  savedPoses.push({
    label,
    pose: normalizePose(currentLandmarks),
    timestamp: new Date().toISOString()
  });

  updateCountsUI();
}

function hasEnoughTrainingData() {
  const good = savedPoses.filter(p => p.label === LABEL_GOOD).length;
  const bad = savedPoses.filter(p => p.label === LABEL_BAD).length;
  return good >= config.minSamplesPerLabel && bad >= config.minSamplesPerLabel;
}

function trainModel() {
  if (!isClassifierReady) {
    setFeedback('AI is nog niet klaar.', 'bad');
    return;
  }

  if (!hasEnoughTrainingData()) {
    const good = savedPoses.filter(p => p.label === LABEL_GOOD).length;
    const bad = savedPoses.filter(p => p.label === LABEL_BAD).length;

    setFeedback(
      `Nog niet genoeg data. Goede: ${good}/${config.minSamplesPerLabel}, Slechte: ${bad}/${config.minSamplesPerLabel}`,
      'bad'
    );
    return;
  }

  const split = makeStratifiedSplit(savedPoses, config.trainSplit);

  trainSet = split.train;
  testSet = split.test;

  if (testSet.length === 0) {
    setFeedback('Kon geen testset maken. Verzamel meer data.', 'bad');
    return;
  }

  knnClassifier = ml5.KNNClassifier();
  for (const p of trainSet) {
    knnClassifier.addExample(p.pose.flat(), p.label);
  }

  hasTrainedModel = true;

  resetMatrixUI();
  setFeedback(`Model getraind. Train: ${trainSet.length}, Test: ${testSet.length}.`, 'good');
}

async function calculateTestMetrics() {
  if (!isClassifierReady || !hasTrainedModel) {
    setFeedback('Train eerst het model voordat je test-accuracy berekent.', 'bad');
    return;
  }

  if (!testSet || testSet.length === 0) {
    setFeedback('Geen testset beschikbaar. Klik eerst op Train Model.', 'bad');
    return;
  }

  const cm = {
    gg: 0,
    gb: 0,
    bg: 0,
    bb: 0
  };

  let correct = 0;

  for (const item of testSet) {
    const actual = item.label;
    const features = item.pose.flat();

    try {
      const pred = await knnClassifier.classify(features);
      const predicted = pred.label;

      if (actual === LABEL_GOOD && predicted === LABEL_GOOD) cm.gg++;
      else if (actual === LABEL_GOOD && predicted === LABEL_BAD) cm.gb++;
      else if (actual === LABEL_BAD && predicted === LABEL_GOOD) cm.bg++;
      else if (actual === LABEL_BAD && predicted === LABEL_BAD) cm.bb++;

      if (actual === predicted) correct++;
    } catch (e) {
      console.error('Classificatie fout bij test:', e);
    }
  }

  const total = testSet.length;
  const acc = total > 0 ? (correct / total) * 100 : 0;

  updateMatrixUI(cm, acc, total);
  setFeedback(`Test accuracy: ${acc.toFixed(1)}% (n=${total}).`, 'neutral');
}

function makeStratifiedSplit(all, trainRatio) {
  const good = all.filter(x => x.label === LABEL_GOOD);
  const bad = all.filter(x => x.label === LABEL_BAD);

  const goodShuffled = shuffleCopy(good);
  const badShuffled = shuffleCopy(bad);

  const goodTrainCount = Math.max(1, Math.floor(goodShuffled.length * trainRatio));
  const badTrainCount = Math.max(1, Math.floor(badShuffled.length * trainRatio));

  const goodTrain = goodShuffled.slice(0, goodTrainCount);
  const goodTest = goodShuffled.slice(goodTrainCount);

  const badTrain = badShuffled.slice(0, badTrainCount);
  const badTest = badShuffled.slice(badTrainCount);

  const train = shuffleCopy([...goodTrain, ...badTrain]);
  const test = shuffleCopy([...goodTest, ...badTest]);

  return { train, test };
}

function shuffleCopy(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
  return a;
}

/* ===========================
   DATA EXPORT / IMPORT / RESET
   =========================== */

function exportToJSON() {
  if (savedPoses.length === 0) {
    setFeedback('Geen data om te exporteren.', 'bad');
    return;
  }

  const jsonStr = JSON.stringify(savedPoses, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `houding-data_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  setFeedback('Export klaar. Check je Downloads map.', 'neutral');
}

function importFromJSON() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json';

  input.onchange = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (!Array.isArray(data)) {
        setFeedback('JSON heeft niet het juiste formaat.', 'bad');
        return;
      }

      const valid = data.every(item =>
        item &&
        (item.label === LABEL_GOOD || item.label === LABEL_BAD) &&
        Array.isArray(item.pose)
      );

      if (!valid) {
        setFeedback('JSON bevat ongeldige poses.', 'bad');
        return;
      }

      savedPoses = data;
      updateCountsUI();

      trainSet = [];
      testSet = [];
      hasTrainedModel = false;
      resetMatrixUI();

      setFeedback(`Data geïmporteerd: ${savedPoses.length} samples. Klik Train Model.`, 'good');
    } catch (err) {
      console.error(err);
      setFeedback('Importeren mislukt. Is dit geldig JSON?', 'bad');
    }
  };

  input.click();
}

function clearData() {
  cancelAllSampling();

  savedPoses = [];
  trainSet = [];
  testSet = [];
  hasTrainedModel = false;

  knnClassifier = ml5.KNNClassifier();
  updateCountsUI();

  liveBadge.textContent = 'Live';
  setBadgeState('neutral');
  liveMeta.textContent = 'Geen voorspelling';

  resetMatrixUI();
  setFeedback('Data gewist. Neem nieuwe samples.', 'neutral');
}

function updateCountsUI() {
  const good = savedPoses.filter(p => p.label === LABEL_GOOD).length;
  const bad = savedPoses.filter(p => p.label === LABEL_BAD).length;

  if (countGoodEl) countGoodEl.textContent = String(good);
  if (countBadEl) countBadEl.textContent = String(bad);
  if (countTotalEl) countTotalEl.textContent = String(savedPoses.length);
}

/* ===========================
   UI HELPERS
   =========================== */

function setFeedback(message, tone) {
  feedbackElement.textContent = message;

  if (tone === 'good') {
    feedbackElement.style.borderColor = 'rgba(74,222,128,0.30)';
  } else if (tone === 'bad') {
    feedbackElement.style.borderColor = 'rgba(251,113,133,0.30)';
  } else {
    feedbackElement.style.borderColor = 'rgba(255,255,255,0.12)';
  }
}

function setStatus(text, state) {
  statusText.textContent = text;

  if (state === 'good') {
    statusDot.style.background = 'var(--good)';
    statusDot.style.boxShadow = '0 0 0 3px rgba(74,222,128,0.15)';
  } else if (state === 'bad') {
    statusDot.style.background = 'var(--bad)';
    statusDot.style.boxShadow = '0 0 0 3px rgba(251,113,133,0.15)';
  } else {
    statusDot.style.background = 'var(--warn)';
    statusDot.style.boxShadow = '0 0 0 3px rgba(251,191,36,0.15)';
  }
}

function setBadgeState(state) {
  if (state === 'good') {
    liveBadge.style.borderColor = 'rgba(74,222,128,0.35)';
    liveBadge.style.background = 'rgba(74,222,128,0.10)';
  } else if (state === 'bad') {
    liveBadge.style.borderColor = 'rgba(251,113,133,0.35)';
    liveBadge.style.background = 'rgba(251,113,133,0.10)';
  } else {
    liveBadge.style.borderColor = 'rgba(255,255,255,0.12)';
    liveBadge.style.background = 'rgba(255,255,255,0.04)';
  }
}

/* ===========================
   CONFUSION MATRIX UI
   =========================== */

function ensureConfusionMatrixUI() {
  if (!accuracyBtn) return;

  const parent = accuracyBtn.parentElement;
  if (!parent) return;

  const existing = document.getElementById('cmatrix');
  if (existing) {
    cmEls = {
      gg: document.getElementById('cmGG'),
      gb: document.getElementById('cmGB'),
      bg: document.getElementById('cmBG'),
      bb: document.getElementById('cmBB'),
      acc: document.getElementById('cmACC'),
      n: document.getElementById('cmN')
    };
    return;
  }

  const wrap = document.createElement('div');
  wrap.id = 'cmatrix';
  wrap.style.marginTop = '12px';
  wrap.style.border = '1px solid rgba(255,255,255,0.12)';
  wrap.style.borderRadius = '12px';
  wrap.style.background = 'rgba(255,255,255,0.03)';
  wrap.style.padding = '12px';

  wrap.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px;">
      <div style="font-size:12px; color: rgba(255,255,255,0.62);">Confusion matrix (Test)</div>
      <div style="font-size:12px; color: rgba(255,255,255,0.62);">Accuracy: <span id="cmACC">—</span> | n=<span id="cmN">—</span></div>
    </div>

    <div style="display:grid; grid-template-columns: 120px 1fr 1fr; gap:8px; align-items:stretch;">
      <div></div>
      <div style="font-size:12px; color: rgba(255,255,255,0.62); text-align:center;">Pred: Goed</div>
      <div style="font-size:12px; color: rgba(255,255,255,0.62); text-align:center;">Pred: Slecht</div>

      <div style="font-size:12px; color: rgba(255,255,255,0.62);">Actual: Goed</div>
      <div style="text-align:center; padding:10px; border:1px solid rgba(255,255,255,0.12); border-radius:10px;" id="cmGG">—</div>
      <div style="text-align:center; padding:10px; border:1px solid rgba(255,255,255,0.12); border-radius:10px;" id="cmGB">—</div>

      <div style="font-size:12px; color: rgba(255,255,255,0.62);">Actual: Slecht</div>
      <div style="text-align:center; padding:10px; border:1px solid rgba(255,255,255,0.12); border-radius:10px;" id="cmBG">—</div>
      <div style="text-align:center; padding:10px; border:1px solid rgba(255,255,255,0.12); border-radius:10px;" id="cmBB">—</div>
    </div>
  `;

  parent.appendChild(wrap);

  cmEls = {
    gg: document.getElementById('cmGG'),
    gb: document.getElementById('cmGB'),
    bg: document.getElementById('cmBG'),
    bb: document.getElementById('cmBB'),
    acc: document.getElementById('cmACC'),
    n: document.getElementById('cmN')
  };
}

function resetMatrixUI() {
  if (!cmEls) return;
  cmEls.gg.textContent = '—';
  cmEls.gb.textContent = '—';
  cmEls.bg.textContent = '—';
  cmEls.bb.textContent = '—';
  cmEls.acc.textContent = '—';
  cmEls.n.textContent = '—';
}

function updateMatrixUI(cm, acc, n) {
  if (!cmEls) return;

  cmEls.gg.textContent = String(cm.gg);
  cmEls.gb.textContent = String(cm.gb);
  cmEls.bg.textContent = String(cm.bg);
  cmEls.bb.textContent = String(cm.bb);

  cmEls.acc.textContent = `${acc.toFixed(1)}%`;
  cmEls.n.textContent = String(n);
} 